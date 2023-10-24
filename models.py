from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch import nn
from enum import IntEnum
import pandas as pd
from typing import Dict

from dataset import BaseDataset, RatingFormat, DatasetRow


class ModelArchitecture(IntEnum):
    NEURAL_CF = 1
    MATRIX_FACTORIZATION = 2
    DEEP_FM = 3
    WIDE_DEEP = 4
    TWO_TOWER = 5


class RecModel(nn.Module):
    def __init__(
        self,
        dataset: BaseDataset,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        embedding_dim: int = 6
        emb_dict = {}

        # List of feature names for each column, used for embeddings
        self.categorical_feature_names: List[str] = dataset.categorical_features.columns.tolist()

        for col_name in dataset.categorical_features.columns:
            emb_dict[col_name] = nn.Embedding(dataset.categorical_feature_sizes[col_name], embedding_dim)
        emb_dict = nn.ModuleDict(emb_dict)
        self.emb_dict = emb_dict
        self.emb_in_size = embedding_dim * len(dataset.categorical_feature_sizes)
        print(f"Created embeddings for features: {dataset.categorical_features.columns.values}")

    def get_feature_embeddings(self, batch: DatasetRow, concat=True) -> Tensor:
        embeddings = []
        for i, feature_name in enumerate(self.categorical_feature_names):
            emb = self.emb_dict[feature_name]
            feature_column = batch.categorical_features[:, i].to(dtype=torch.int64, device=self.device)
            embedded_column = emb(feature_column)
            embeddings.append(embedded_column)
        embeddings = torch.stack(embeddings, dim=1).squeeze()
        if concat:
            embeddings = embeddings.view(-1, self.emb_in_size)
        return embeddings

    def create_linear_tower(self, layers: List[int]) -> nn.Sequential:
        """
        Creates a linear tower with ReLU activations and dropout, with the embedding size as input
        and the output size as the last element of the layers list
        """
        all_layers = []
        # insert the embedding size as the first element
        layers.insert(0, self.emb_in_size)
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            all_layers.append(nn.Linear(in_size, out_size))
            all_layers.append(nn.BatchNorm1d(out_size))
            all_layers.append(nn.ReLU())
            # all_layers.append(nn.Dropout(0.1))
        return nn.Sequential(*all_layers)

class WideDeepModel(RecModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.wide_layer = nn.Linear(self.emb_in_size, 1)
        layers = [64, 32, 16, 8, 1]

        self.deep_layers = self.create_linear_tower(layers)

    def forward(self, batch: DatasetRow):
        emb_cat = self.get_feature_embeddings(batch)
        wide_out = self.wide_layer(emb_cat)
        deep_out = self.deep_layers(emb_cat)
        x = wide_out + deep_out
        x = torch.sigmoid(x)
        return x

def get_fm_loss(embeddings: torch.Tensor):
    square_of_sum = torch.sum(embeddings, dim=1) ** 2
    sum_of_square = torch.sum(embeddings ** 2, dim=1)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * torch.sum(cross_term, dim=1)
    return cross_term


class DeepFMModel(RecModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = [64, 32, 16, 8, 1]

        # used to compute the FM loss
        self.fc_layers = self.create_linear_tower(layers)
        self.lin = nn.Linear(self.emb_in_size, 1)

    def forward(self, batch):
        emb = self.get_feature_embeddings(batch, concat=False)
        emb_concat = emb.view(-1, self.emb_in_size)
        fm_loss = get_fm_loss(emb)
        fc_loss = self.fc_layers(emb_concat).squeeze()
        lin_loss = self.lin(emb_concat).squeeze()
        loss = lin_loss + fc_loss + fm_loss
        return loss


class MatrixFactorizationModel(RecModel):
    def forward(self, batch):
        embeddings = self.get_feature_embeddings(batch, concat=False)
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings_prod = torch.prod(embeddings, dim=1)
        interaction = torch.sum(embeddings_prod, dim=1)
        return interaction


class NeuralCFModel(RecModel):
    def __init__(
        self,
        dataset: BaseDataset,
        device: torch.device,
    ):
        super().__init__(dataset, device)
        layers: List[int] = [ 128, 64, 32, 16, 8, 4, 1]
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        self.bn = nn.BatchNorm1d(self.emb_in_size)

        self.fc_layers = self.create_linear_tower(layers)

    def forward(self, batch):
        x = self.get_feature_embeddings(batch)
        x = self.bn(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc_layers(x)
        return x

class TwoTowerModel(RecModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = [64, 32, 16, 8, 1]

        self.user_tower = self.create_linear_tower(layers)
        self.item_tower = self.create_linear_tower(layers)

    def forward(self, batch):
        emb_cat = self.get_feature_embeddings(batch)
        user_out = self.user_tower(emb_cat)
        item_out = self.item_tower(emb_cat)
        x = user_out + item_out
        x = torch.sigmoid(x)
        return x



models_dict: Dict[ModelArchitecture, type[RecModel]] = {
    ModelArchitecture.MATRIX_FACTORIZATION: MatrixFactorizationModel,
    ModelArchitecture.NEURAL_CF: NeuralCFModel,
    ModelArchitecture.DEEP_FM: DeepFMModel,
    ModelArchitecture.WIDE_DEEP: WideDeepModel,
    ModelArchitecture.TWO_TOWER: TwoTowerModel,
}
