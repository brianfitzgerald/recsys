import numpy as np
import pandas as pd
import torch.utils.data
from enum import IntEnum
from numpy.random import choice, randint


class RatingFormat(IntEnum):
    BINARY = 1
    RATING = 2


class MovieLens20MDataset(torch.utils.data.Dataset):

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

    def __init__(self, dataset_path: str, return_format: RatingFormat, negative_sample_threshold: float = 3, num_negative_samples: int = 4, max_users: int = None):
        data = pd.read_csv(
            dataset_path, sep=",", engine="c", header="infer", nrows=10000
        ).to_numpy()[:, :3]
        self.data = data

        # if max_users is not None:
        #     first_n_users = np.unique(data[:, 0])[:max_users]
        #     ratings_from_first_n_users = data[np.where(np.isin(data[:, 0], first_n_users))]
        #     data = ratings_from_first_n_users


        self.no_users = np.max(data[:, 0].astype(np.int64)) + 1
        self.no_movies = np.max(data[:, 1].astype(np.int64)) + 1
        self.no_samples = len(data)
        self.num_negative_samples = num_negative_samples
        print(f"Number of users: {self.no_users} | Number of movies: {self.no_movies} | Number of samples: {self.no_samples}")

        self.positive_samples = data[data[:, 2] >= negative_sample_threshold]
        self.negative_samples = data[data[:, 2] < negative_sample_threshold]

        pos_ratings_mask = np.zeros(len(self.positive_samples), dtype=np.int64)
        pos_ratings_mask[::num_negative_samples] = 1
        
        self.return_format = return_format
        self.neg_threshold = negative_sample_threshold

    def __len__(self):
        return self.no_samples

    def __getitem__(self, index):
        sample = self.data[index]
        rating = sample[2].astype(np.float32)
        if self.return_format == RatingFormat.BINARY:
            rating = rating >= self.neg_threshold
        return sample[0].astype(np.int64), sample[1].astype(np.int64), rating.astype(np.float32)
        sample_positive = index % self.num_negative_samples != 0
        sample_idx = np.random.choice(self.positive_samples.shape[0]) if sample_positive else np.random.choice(self.negative_samples.shape[0])
        sample = self.positive_samples[sample_idx] if sample_positive else self.negative_samples[sample_idx]
        rating = sample[2].astype(np.float32)
        # print(f"Sample: {rating} | {index % self.num_negative_samples} | {sample_positive}")
        return sample[0].astype(np.int64), sample[1].astype(np.int64), rating.astype(np.float32)
