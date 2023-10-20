from collections import defaultdict
from enum import IntEnum
from typing import Optional

import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset

from utils import get_available_device

pd.options.display.float_format = "{:.2f}".format

import fire
import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import (BaseDataset, DatasetSource, MovieLens20MDataset,
                     RatingFormat, datasets_dict)
from metrics import *
from models import *

torch.manual_seed(0)


class Params:
    learning_rate: int = 5e-3
    weight_decay: float = 1e-5

    embedding_dim: int = 32
    dropout: float = 0.2
    batch_size: int = 32
    eval_size: int = 100
    max_rows: Optional[int] = None
    model_architecture: ModelArchitecture = ModelArchitecture.TWO_TOWER
    dataset_source: DatasetSource = DatasetSource.CRITEO
    rating_format: RatingFormat = RatingFormat.BINARY
    max_users: Optional[int] = None
    num_epochs: int = 100

    do_eval: bool = False
    eval_every: int = 1
    max_batches: int = 10

    @classmethod
    def default_values(cls):
        instance = cls()
        attrs_dict = {
            attr: getattr(instance, attr)
            for attr in dir(instance)
            if not callable(getattr(instance, attr)) and not attr.startswith("__")
        }
        for key, value in attrs_dict.items():
            if isinstance(value, IntEnum):
                attrs_dict[key] = value.name
        return attrs_dict


class RecommenderModule(nn.Module):
    def __init__(self, recommender: RecModel, use_wandb: bool):
        super().__init__()
        self.recommender = recommender
        if (
            Params.model_architecture == ModelArchitecture.MATRIX_FACTORIZATION
        ):
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = torch.nn.BCELoss()
        self.use_wandb = use_wandb

    def training_step(self, batch: DatasetRow):
        preds = self.recommender(batch).squeeze()
        labels = batch.labels.to(dtype=torch.float32, device=preds.device)
        loss = self.loss_fn(preds, labels)
        # print(f"Loss: {loss.item():03.3f} preds: {preds.tolist()} ratings: {ratings.tolist()}")
        if self.use_wandb:
            wandb.log({"train_loss": loss})
        return loss

    def eval_step(self, dataset: BaseDataset, batch: DatasetRow, k: int = 10):
        with torch.no_grad():
            users, items = dataset.categorical_features["userId"], dataset.categorical_features["movieId"]
            max_user_id = int(users.max().item() + 1)
            preds = self.recommender(batch).squeeze()
            # TODO rewrite
            eval_loss = self.loss_fn(preds, dataset.labels).item()
            user_item_ratings = np.empty((max_user_id, k))
            true_item_ratings = np.empty((max_user_id, k))
            for i, user_id in enumerate(users):

                user_id = user_id.int().item()
                # predict every item for every user
                user_ids = torch.full_like(items, user_id)
                user_batch = torch.stack([user_ids, items], dim=1)
                user_preds = self.recommender((user_batch, None)).squeeze()
                top_k_preds = torch.topk(user_preds, k=k).indices
                user_item_ratings[user_id] = top_k_preds.numpy()

                true_top_k = torch.topk(ratings, k=k).indices
                true_item_ratings[user_id] = true_top_k.numpy()
                if i == 0:
                    dataset.display_recommendation_output(
                        user_id, top_k_preds, true_top_k
                    )

            unique_item_catalog = list(set(items.tolist()))
            item_popularity = defaultdict(int)
            for item in items:
                item_popularity[item.item()] += 1

            num_users = len(list(set(users.tolist())))
            num_items = len(list(set(items.tolist())))

            novelty = novelty_score(
                user_item_ratings, item_popularity, num_users, num_items
            )

            user_rating_preds = np.array(
                [p for sublist in user_item_ratings for p in sublist]
            )
            user_rating_ref = np.array(
                [p for sublist in user_item_ratings for p in sublist]
            )

            prediction_coverage = prediction_coverage_score(
                user_item_ratings, unique_item_catalog
            )
            catalog_coverage = catalog_coverage_score(
                user_item_ratings, unique_item_catalog, k
            )

            personalization = personalization_score(user_item_ratings)

            ref_bool, preds_bool = user_rating_ref.astype(
                bool
            ), user_rating_preds.astype(bool)
            # Handle the case where all values are T or F
            if len(np.unique(ref_bool)) == 2 and len(np.unique(preds_bool)) == 2:
                roc_auc = roc_auc_score(ref_bool, preds_bool)

            # gives the index of the top k predictions for each sample
            log_dict = {
                "eval_loss": eval_loss,
                "ndcg": ndcg_score(user_rating_preds, user_rating_ref),
                "novelty": novelty,
                "prediction_coverage": prediction_coverage,
                "catalog_coverage": catalog_coverage,
                "personalization": personalization,
                "roc_auc": roc_auc,
            }

            print(log_dict)
            if self.use_wandb:
                wandb.log(log_dict)


def main(
    use_wandb: bool = False,
):
    device = get_available_device()

    dataset: BaseDataset = datasets_dict[Params.dataset_source]()
    print(f"Loading dataset {dataset.__class__.__name__}..")

    train_size = len(dataset) - Params.eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, Params.eval_size]
    )
    # Multithreaded dataloading seems to be slower when using MPS
    train_dataloader = DataLoader(
        train_dataset, batch_size=Params.batch_size, shuffle=True, num_workers=0
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=Params.eval_size, shuffle=True, num_workers=0
    )
    model_cls: RecModel = models_dict[Params.model_architecture]
    print(f"Loading model {model_cls.__name__}..")
    model: RecModel = model_cls(
        dataset,
        device
    ).to(device)
    model.train()
    module = RecommenderModule(model, use_wandb).to(device)
    if use_wandb:
        wandb.init(project="recsys", config=Params.default_values())
        wandb.watch(model)
    optimizer = AdamW(
        module.parameters(), lr=Params.learning_rate, weight_decay=Params.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Params.num_epochs, eta_min=1e-6)
    print("Starting training...")
    for i in range(Params.num_epochs):
        if i % Params.eval_every == 0 and Params.do_eval:
            print("Running eval..")
            for j, batch in enumerate(eval_dataloader):
                module.eval_step(dataset, batch, 10)
                break
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
            learning_rate = scheduler.get_last_lr()[0]
            if use_wandb:
                wandb.log({"total_norm": total_norm.item(), "lr": learning_rate})


            print(
                f"Epoch {i:03.0f}, batch {j:03.0f}, loss {loss.item():03.3f}, total norm: {total_norm.item():03.3f}, lr {learning_rate:03.5f}"
            )

            if j > Params.max_batches:
                break

            # if Params.model_architecture != ModelArchitecture.MATRIX_FACTORIZATION:
            #     torch.nn.utils.clip_grad_norm_(module.parameters(), 100)
        scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
