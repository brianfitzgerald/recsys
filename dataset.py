import numpy as np
import pandas as pd
import torch.utils.data

class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.user_ids = data[:, 0].astype(np.int)
        self.movie_ids = data[:, 1].astype(np.int)
        self.ratings = data[:, 2].astype(np.float32)
        self.no_users = np.max(self.user_ids) + 1
        self.no_movies = np.max(self.movie_ids) + 1
        self.user_field_idx = np.array((0, ), dtype=np.int)
        self.item_field_idx = np.array((1,), dtype=np.int)

    def __len__(self):
        return self.movie_ids.shape[0]

    def __getitem__(self, index):
        return self.user_ids[index], self.movie_ids[index], self.ratings[index]
