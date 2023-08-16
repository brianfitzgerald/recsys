import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import CSVLogger
import torchvision.transforms as transforms
import fire

from dataset import MovieLens20MDataset


torch.manual_seed(0)


class Recommender(nn.Module):
    def __init__(self, n_users, n_movies, layers=[16, 8, 4, 2], dropout: float = 0):
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        embedding_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.dropout = dropout

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 1 is the output dimension
        self.last_linear = torch.nn.Linear(layers[-1], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.movie_embedding(items)
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        logit = self.last_linear(x)
        rating = self.sigmoid(logit)
        return rating


class RecommenderModule(pl.LightningModule):
    def __init__(self, recommender: Recommender):
        super().__init__()
        self.recommender = recommender
        self.loss_fn = torch.nn.MSELoss()
        self.save_hyperparameters()

    def training_step(self, batch):
        users, items, ratings = batch
        preds = self.recommender(users, items).squeeze(1)
        ratings = ratings / 5
        loss = self.loss_fn(preds, ratings)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        users, items, ratings = batch
        preds = self.recommender(users, items).squeeze(1)
        ratings = ratings / 5
        loss = self.loss_fn(preds, ratings)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

def main(use_wandb: bool = False):

    dataset = MovieLens20MDataset("ml-25m/ratings.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    no_users, no_movies = dataset.no_movies, dataset.no_users
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=512)
    val_dataloader = DataLoader(test_dataset, batch_size=512)
    model = RecommenderModule(Recommender(no_movies, no_users))
    logger = None
    if use_wandb:
        logger = WandbLogger(project="recsys")
        logger.watch(model)
    else:
        logger = None
    trainer = pl.Trainer(logger=logger, precision=64)
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

if __name__ == "__main__":
    fire.Fire(main)