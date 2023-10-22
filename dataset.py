import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from enum import IntEnum
import os
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from typing import NamedTuple, Optional, Dict, Union


class RatingFormat(IntEnum):
    BINARY = 1
    RATING = 2

class DatasetRow(NamedTuple):
    dense_features: Optional[Tensor]
    categorical_features: Tensor
    labels: Tensor


class BaseDataset(Dataset):
    def __init__(self, categorical_features: pd.DataFrame, dense_features: pd.DataFrame, labels: pd.Series) -> None:
        super().__init__()

        # categorical features are converted to embeddings
        # (num_features,)
        self.categorical_features: pd.DataFrame = categorical_features
        # (num_features,)
        self.categorical_feature_sizes: pd.Series = categorical_features.max() + 1
        self.categorical_feature_names=self.categorical_features.columns.tolist()
        self.dense_features: pd.DataFrame = dense_features
        self.labels: Tensor = torch.tensor(labels.values)

        assert self.categorical_features.shape[0] == self.dense_features.shape[0]

    def __len__(self):
        return self.categorical_features.shape[0]

    def display_recommendation_output(
        self, user_id: int, pred_ids: np.ndarray, true_ids: np.ndarray
    ):
        """Displays a table of recommendations for a certain user"""
        raise NotImplementedError("Please implement a recommendation output fn for this dataset!")

    def __getitem__(self, index) -> DatasetRow:
        return DatasetRow(
            dense_features=torch.tensor(self.dense_features.loc[index].values),
            categorical_features=torch.tensor(self.categorical_features.loc[index].values),
            labels=self.labels[index]
        )

class MovieLens20MDataset(BaseDataset):
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

    def __init__(
        self,
        dataset_dir: str = "datasets/ml-25m",
        max_users: Optional[int] = None,
    ):

        ratings_data = pd.read_csv(
            os.path.join(dataset_dir, "ratings.csv"),
            sep=",",
            header="infer",
            engine="pyarrow",
        ).dropna()

        self.neg_threshold = 2.5
        ratings_data["rating"] = ratings_data["rating"].apply(
            lambda x: 1 if x >= self.neg_threshold else 0
        )

        movie_data = pd.read_csv(
            os.path.join(dataset_dir, "movies.csv"),
            sep=",",
            engine="pyarrow",
            header="infer",
        )

        primary_genre_per_movie = movie_data["genres"].str.split("|").str[0]
        self.movie_genres = pd.concat([movie_data["movieId"], primary_genre_per_movie])

        if max_users is not None:
            first_n_users = ratings_data["userId"].unique()[:max_users]
            ratings_from_first_n_users = ratings_data[
                ratings_data["userId"].isin(first_n_users)
            ]
            ratings_data = ratings_from_first_n_users

        categorical_features = ["userId", "movieId"]
        dense_features = ["timestamp"]

        no_users = ratings_data["userId"].max()
        no_movies = ratings_data["movieId"].max()
        no_samples = ratings_data.shape[0]

        print(
            f"Number of users: {no_users} | Number of movies: {no_movies} | Number of samples: {no_samples}"
        )

        super().__init__(ratings_data[categorical_features], ratings_data[dense_features], ratings_data["rating"])

    def display_recommendation_output(
        self, user_id: int, pred_ids: np.ndarray, true_ids: np.ndarray
    ):
        """Displays a table of recommendations for a certain user"""
        pred_data = self.categorical_features.iloc[pred_ids]
        true_data = self.categorical_features.iloc[true_ids]
        # print(f"predictions for user {user_id}:")
        # print(tabulate(pred_data[["title", "genres"]], headers="keys", tablefmt="psql"))
        # print(f"ground truth for user {user_id}:")
        # print(tabulate(true_data[["title", "genres"]], headers="keys", tablefmt="psql"))


class CriteoDataset(BaseDataset):

    def __init__(self) -> None:

        continuous_variable = [f"I{i}" for i in range(1, 14)]
        discrete_variable = [f"C{i}" for i in range(1, 27)]
        columns = ["label"] + continuous_variable + discrete_variable
        all_data = pd.read_csv("datasets/criteo_1m.txt", sep='\t', names=columns)

        # print(f"Missing values per column: {all_data.isnull().sum()}")
        print(f"Label counts: {all_data['label'].value_counts()}")

        # breakpoint()
        all_data.fillna(0, inplace=True)


        labeler = LabelEncoder()

        categorical_features = all_data.loc[:, ["C1", "C2", "C5", "C7"]]
        dense_features = all_data.iloc[::,1:14]
        labels = all_data["label"]

        for feature in categorical_features.columns:
            categorical_features[feature] = labeler.fit_transform(categorical_features[feature])
        
        assert categorical_features.shape[0] == dense_features.shape[0] == labels.shape[0]
        super().__init__(categorical_features, dense_features, labels)

class DatasetSource(IntEnum):
    MOVIELENS = 1
    AMAZON = 2
    CRITEO = 3
    SPOTIFY = 4


datasets_dict: Dict[DatasetSource, type[Union[MovieLens20MDataset, CriteoDataset]]] = {
    DatasetSource.CRITEO: CriteoDataset,
    DatasetSource.MOVIELENS: MovieLens20MDataset,
}