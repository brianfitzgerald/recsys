import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl

torch.manual_seed(0)


class Recommender(nn.Module):
    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False):
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        embedding_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 1 is the output dimension
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

class RecommenderModule(pl.LightningModule):
    def __init__(self, recommender: Recommender):
        super().__init__()
        self.recommender = recommender
        self.loss_fn = torch.nn.BCELoss()
    
    def training_step(self, batch):
        users = batch["user_id"]
        items = batch["item_id"]
        rating = batch["rating"]
        x = self.recommender(users, items)
        loss = self.loss_fn(x, rating)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer