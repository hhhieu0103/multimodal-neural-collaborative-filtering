import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        users = df.iloc[:, 0].values
        items = df.iloc[:, 1].values
        ratings = df.iloc[:, 2].values
        
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return self.users.size(0)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]