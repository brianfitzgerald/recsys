from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import fire
import wandb

from dataset import MovieLens20MDataset, RatingFormat


torch.manual_seed(0)


class Params:
    learning_rate: int = 1e-2
    layers: List[int] = [16, 8]
    dropout: float = 0.2
    batch_size: int = 2048
    weight_decay: float = 1e-5
    rating_format: RatingFormat = RatingFormat.BINARY


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
        logit = self.output_layer(x)
        if Params.rating_format == RatingFormat.BINARY:
            rating = torch.sigmoid(logit)
        return rating


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

    def eval_step(self, batch):
        with torch.no_grad():
            users, items, ratings = batch
            preds = self.recommender(users, items).squeeze(1)
            loss = self.loss_fn(preds, ratings)
            if self.use_wandb:
                wandb.log({"eval_loss": loss})
            return loss


def main(
    use_wandb: bool = False,
    num_epochs: int = 5000,
    eval_every: int = 10000,
    max_batches: int = 10000,
):
    dataset = MovieLens20MDataset("ml-25m/ratings.csv", Params.rating_format)
    test_size = 1000
    train_size = len(dataset) - test_size
    no_users, no_movies = dataset.no_movies, dataset.no_users
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=Params.batch_size)
    eval_dataloader = DataLoader(test_dataset, batch_size=Params.batch_size)
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

            torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
            optimizer.step()
            if j % eval_every == 0:
                print("Running eval..")
                for j, batch in enumerate(eval_dataloader):
                    eval_loss = module.eval_step(batch)
                    print(f"Eval loss for batch {j}: {eval_loss.item()}")
                    break


if __name__ == "__main__":
    fire.Fire(main)
