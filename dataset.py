import torch
import pandas as pd
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(
            self,
            df_interaction: pd.DataFrame,
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            rating_col: str = 'rating_imp',
    ):
        """
        Dataset for NCF model with support for time features and metadata.

        Args:
            df_interaction: Interaction dataframe with user_idx, item_idx, and rating columns
        """
        users = df_interaction[user_col].values
        items = df_interaction[item_col].values
        ratings = df_interaction[rating_col].values

        # Ensure indices are properly cast as integers
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        """Get the number of interactions in the dataset"""
        return self.users.size(0)

    def __getitem__(self, idx):
        """Get a single interaction with its features"""

        return self.users[idx], self.items[idx], self.ratings[idx]