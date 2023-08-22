import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
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


class RecommenderModule(nn.Module):
    def __init__(self, recommender: Recommender):
        super().__init__()
        self.recommender = recommender
        self.loss_fn = torch.nn.MSELoss()

    def training_step(self, batch):
        users, items, ratings = batch
        preds = self.recommender(users, items).squeeze(1)
        ratings = ratings / 5
        loss = self.loss_fn(preds, ratings)
        return loss.float()

    def eval_step(self, batch):
        with torch.no_grad():
            users, items, ratings = batch
            preds = self.recommender(users, items).squeeze(1)
            ratings = ratings / 5
            loss = self.loss_fn(preds, ratings)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

def main(use_wandb: bool = False, num_epochs: int = 5000, eval_every: int = 10):

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    for i in range(num_epochs):
        for j, batch in enumerate(train_dataloader):
            loss = model.training_step(batch)
            loss.backward()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm()

            print("Total norm: ", total_norm)
            print(f"Epoch {i}, batch {j}, loss {loss.item()}, total norm: {total_norm}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
        if i % eval_every == 0:
            print("Running eval..")
            for j, batch in enumerate(val_dataloader):
                eval_loss = model.eval_step(batch, i)
                print(f"Eval loss for batch {j}: {eval_loss.item()}")


if __name__ == "__main__":
    fire.Fire(main)