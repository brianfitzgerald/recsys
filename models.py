from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from enum import IntEnum
import pandas as pd

from dataset import RatingFormat


class ModelArchitecture(IntEnum):
    NEURAL_CF = 1
    MATRIX_FACTORIZATION = 2
    DEEP_FM = 3
    WIDE_DEEP = 4


def create_embedding_dict(
    feature_counts: List[int], emb_columns: List[str], embedding_dim: int
) -> Tuple[nn.ModuleDict, int]:
    emb_dict = {}
    for i, col_name in enumerate(emb_columns):
        emb_dict[col_name] = nn.Embedding(feature_counts[i], embedding_dim)
    emb_dict = nn.ModuleDict(emb_dict)
    return emb_dict, embedding_dim * len(emb_columns)

def concat_embeddings(input, emb_dict: nn.ModuleDict):
    embeddings = []
    for v in emb_dict.values():
        embeddings.append(v)
    return torch.cat(embeddings, 1)

class WideDeepModel(nn.Module):
    def __init__(
        self, emb_dict: nn.ModuleDict, linear_in_size: int, layers: List[int]
    ) -> None:
        super().__init__()
        
        self.fc_layers = torch.nn.ModuleList()
        self.embeddings = create_embedding_dict(emb_dict)

        self.linear_layer = torch.nn.Linear(linear_in_size, 1)

        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(0.1))

    def forward(self, users, items):
        emb_cat = concat_embeddings(emb_dict)
        x = x + self.linear_layer(emb_cat)
        x = self.fc_layers(x)
        x = x + get_fm_loss(emb_cat)
        x = torch.sigmoid(x)
        return x


class DeepFMModel(nn.Module):
    def __init__(
        self, n_users: int, n_movies: int, embedding_dim: int, layers: List[int]
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_movies, embedding_dim)
        self.fc_layers = torch.nn.ModuleList()

        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(0.1))

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        emb_cat = torch.cat([user_emb, item_emb], 1)
        x = self.fc_layers(x)
        x = x + get_fm_loss(emb_cat)
        x = torch.sigmoid(x)
        return x


def get_fm_loss(emb_cat: torch.Tensor):
    square_of_sum = torch.pow(torch.sum(emb_cat, dim=1, keepdim=True), 2)
    sum_of_square = torch.sum(emb_cat * emb_cat, dim=1, keepdim=True)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

    return cross_term


class MatrixFactorizationModel(nn.Module):
    def __init__(self, n_users: int, n_movies: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_movies, embedding_dim)

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        interaction = torch.sum(user_emb * item_emb, dim=1)
        interaction = torch.sigmoid(interaction)
        return interaction


class NeuralCFModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_movies: int,
        layers: List[int],
        dropout: float,
        rating_format: RatingFormat,
    ):
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        embedding_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_movies, embedding_dim)
        self.dropout = dropout
        self.rating_format = rating_format

        self.bn = nn.BatchNorm1d(layers[0])

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 1 is the output dimension
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        x = torch.cat([user_emb, item_emb], 1)
        x = self.bn(x)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        rating = self.output_layer(x)
        if self.rating_format == RatingFormat.BINARY:
            rating = torch.sigmoid(rating)
        return rating.squeeze(1).float()
