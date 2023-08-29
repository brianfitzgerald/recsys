import numpy as np
import pandas as pd
import torch.utils.data
from enum import IntEnum


class ReturnFormat(IntEnum):
    BINARY = 1
    RATING = 2


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    # tags.csv:
    # userId,movieId,tag,timestamp

    # ratings.csv:
    # userId,movieId,tag,timestamp

    # movies.csv:
    # movieId,title,genres

    # genome-tags.csv:
    # tagId,tag

    # genome-scores.csv:
    # movieId,tagId,relevance

    def __init__(self, dataset_path: str, return_format: ReturnFormat):
        data = pd.read_csv(
            dataset_path, sep=",", engine="c", header="infer"
        ).to_numpy()[:, :3]
        self.user_ids = data[:, 0].astype(np.int)
        self.movie_ids = data[:, 1].astype(np.int)
        self.ratings = data[:, 2].astype(np.float32)
        self.no_users = np.max(self.user_ids) + 1
        self.no_movies = np.max(self.movie_ids) + 1
        # options: "rating", "binary"
        self.return_format = return_format

    def __len__(self):
        return self.movie_ids.shape[0]

    def __getitem__(self, index):
        rating = (
            self.ratings[index] / 5
            if self.return_format == ReturnFormat.RATING
            else self.ratings[index] >= 3
        ).astype(np.float32)
        return self.user_ids[index], self.movie_ids[index], rating
