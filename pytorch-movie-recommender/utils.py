import pandas as pd
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings["user_id"].values
        self.items = ratings["movie_id"].values
        self.labels = ratings["rating"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

def load_data():
    ratings = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )
    movies = movies[[0, 1]]
    movies.columns = ["movie_id", "title"]

    return ratings, movies
