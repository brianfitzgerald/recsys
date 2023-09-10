from math import log2
import random
from typing import List, Optional
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import fire
import wandb
import numpy as np
from collections import defaultdict

from dataset import MovieLens20MDataset, RatingFormat


torch.manual_seed(0)


class Params:
    learning_rate: int = 1e-3
    weight_decay: float = 1e-5
    layers: List[int] = [64, 32, 16, 8]
    dropout: float = 0.2
    batch_size: int = 256
    rating_format: RatingFormat = RatingFormat.RATING
    max_users: Optional[int] = None
    max_rows: int = 100000


class Recommender(nn.Module):
    def __init__(self, n_users, n_movies, layers, dropout):
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        embedding_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.dropout = dropout

        self.bn = nn.BatchNorm1d(layers[0])

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 1 is the output dimension
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.movie_embedding(items)
        x = torch.cat([user_embedding, item_embedding], 1)
        x = self.bn(x)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        rating = self.output_layer(x)
        if Params.rating_format == RatingFormat.BINARY:
            rating = torch.sigmoid(rating)
        return rating


def ndcg_score(y_true, y_score):
    rating_pairs = np.stack([y_true, y_score], axis=1).tolist()
    sorted_pairs = sorted(rating_pairs, key=lambda x: x[1], reverse=True)
    dcg = sum(
        (true_rating / log2(index + 2))
        for index, (true_rating, _) in enumerate(sorted_pairs)
    )
    ideal_pairs = sorted(rating_pairs, key=lambda x: x[0], reverse=True)
    idcg = sum(
        (true_rating / log2(index + 2))
        for index, (true_rating, _) in enumerate(ideal_pairs)
    )
    ndcg = dcg / idcg
    return ndcg


def novelty_score(predicted: List[int], pop: List[int], num_users: int, num_items: int):
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            if pop[i] > 0:
                self_information += np.sum(-np.log2(pop[i] / num_users))
            else:
                continue
        mean_self_information.append(self_information / num_items)
    novelty = sum(mean_self_information) / k
    return novelty

def prediction_coverage(predicted: List[list], catalog: list):
    unique_items_catalog = set(catalog)
    if len(catalog)!=len(unique_items_catalog):
        raise AssertionError("Duplicated items in catalog")

    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_items_pred = set(predicted_flattened)
    
    if not unique_items_pred.issubset(unique_items_catalog):
        raise AssertionError("There are items in predictions but unseen in catalog.")
    
    num_unique_predictions = len(unique_items_pred)
    prediction_coverage = round(num_unique_predictions/(len(catalog)* 1.0)* 100, 2)
    return prediction_coverage

def catalog_coverage(predicted: List[list], catalog: list, k: int) -> float:
    sampling = random.choices(predicted, k=k)
    predicted_flattened = [p for sublist in sampling for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions/(len(catalog)*1.0)*100,2)
    return catalog_coverage


class RecommenderModule(nn.Module):
    def __init__(self, recommender: Recommender, use_wandb: bool):
        super().__init__()
        self.recommender = recommender
        if Params.rating_format == RatingFormat.BINARY:
            self.loss_fn = torch.nn.BCELoss()
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.use_wandb = use_wandb

    def training_step(self, batch):
        users, items, ratings = batch
        preds = self.recommender(users, items).squeeze(1).float()
        loss = self.loss_fn(preds, ratings)
        if self.use_wandb:
            wandb.log({"train_loss": loss})
        return loss

    def eval_step(self, batch, k: int = 10):
        with torch.no_grad():
            users, items, ratings = batch
            preds = self.recommender(users, items).squeeze(1)
            eval_loss = self.loss_fn(preds, ratings).item()
            user_item_ratings = []
            for user_id in users:
                user_id = user_id.item()
                # predict every item for every user
                user_ids = torch.full_like(items, user_id)
                user_preds = self.recommender(user_ids, items).squeeze(1)
                top_k_preds = torch.topk(user_preds, k=len(items)).indices
                user_item_ratings.append(top_k_preds.tolist())

            item_popularity = defaultdict(int)
            for item in items:
                item_popularity[item.item()] += 1

            novelty = novelty_score(
                user_item_ratings, item_popularity, len(users), len(items)
            )

            # gives the index of the top k predictions for each sample
            log_dict = {
                "eval_loss": eval_loss,
                "ndcg": ndcg_score(ratings, preds),
                "novelty": novelty,
                "prediction_coverage": prediction_coverage(user_item_ratings, item_popularity.keys()),
                "catalog_coverage": catalog_coverage(user_item_ratings, item_popularity.keys(), k),
            }

            print(log_dict)
            if self.use_wandb:
                wandb.log(log_dict)


def main(
    use_wandb: bool = False,
    num_epochs: int = 5000,
    eval_every: int = 100,
    max_batches: int = 100,
    eval_size: int = 1000,
):
    print("Loading dataset..")
    dataset = MovieLens20MDataset(
        "ml-25m/ratings.csv", Params.rating_format, Params.max_rows, Params.max_users
    )
    train_size = len(dataset) - eval_size
    no_users, no_movies = dataset.no_movies, dataset.no_users
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=Params.batch_size, shuffle=False, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_size, shuffle=False)
    model = Recommender(
        no_movies, no_users, layers=Params.layers, dropout=Params.dropout
    )
    model.train()
    module = RecommenderModule(model, use_wandb)
    if use_wandb:
        wandb.init(project="recsys")
        wandb.watch(model)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=Params.learning_rate, weight_decay=Params.weight_decay
    )
    for i in range(num_epochs):
        for j, batch in enumerate(train_dataloader):
            loss = module.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            grads = [
                param.grad.detach().flatten()
                for param in module.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm()
            if use_wandb:
                wandb.log({"total_norm": total_norm.item()})

            print(
                f"Epoch {i:03.0f}, batch {j:03.0f}, loss {loss.item():03.3f}, total norm: {total_norm.item():03.3f}"
            )

            if j > max_batches:
                break

            torch.nn.utils.clip_grad_norm_(module.parameters(), 100)
            if j % eval_every == 0:
                print("Running eval..")
                for j, batch in enumerate(eval_dataloader):
                    module.eval_step(batch, eval_size)
                    break


if __name__ == "__main__":
    fire.Fire(main)
