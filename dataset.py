import torch
import pandas as pd
from torch.utils.data import Dataset
from helpers.mem_map_dataloader import MemMapDataLoader

class NCFDataset(Dataset):
    def __init__(
            self,
            df_interaction: pd.DataFrame,
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            rating_col: str = 'rating_imp',
            feature_dims = None, # Dictionary, key: feature, value: (input, output)
            df_features = None,
            image_dataloader: MemMapDataLoader=None,
            audio_dataloader: MemMapDataLoader=None,
    ):
        """
        Dataset for NCF model with support for time features and metadata.

        Args:
            df_interaction: Interaction dataframe with user_idx, item_idx, and rating columns
        """
        users = df_interaction[user_col].values
        items = df_interaction[item_col].values
        ratings = df_interaction[rating_col].values

        self.users = torch.tensor(users, dtype=torch.int)
        self.items = torch.tensor(items, dtype=torch.int)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        self.feature_dims = feature_dims
        self.df_features = df_features.copy().set_index('item_id') if df_features is not None else None

        self.image_dataloader = image_dataloader
        self.audio_dataloader = audio_dataloader

    def __len__(self):
        """Get the number of interactions in the dataset"""
        return self.users.size(0)

    def __getitem__(self, idx):
        """Get a single interaction with its features"""
        features_idx = None
        if self.df_features is not None and self.feature_dims is not None:
            features_idx = {}
            item_id = self.items[idx].item()
            for feature, (input_dim, output_dim) in self.feature_dims.items():
                feature_value = self.df_features.loc[item_id][feature]

                if input_dim == 1:
                    features_idx[feature] = torch.tensor(feature_value, dtype=torch.float32)
                else:
                    features_idx[feature] = torch.tensor(feature_value, dtype=torch.int)

        image_tensor = None
        if self.image_dataloader is not None:
            item_idx = self.items[idx].item()
            image_features = self.image_dataloader.get_tensor(item_idx)
            image_tensor = torch.tensor(image_features, dtype=torch.float32)

        audio_tensor = None
        if self.audio_dataloader is not None:
            item_idx = self.items[idx].item()
            audio_tensor = self.audio_dataloader.get_tensor(item_idx)
            if not isinstance(audio_tensor, torch.Tensor):
                audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

        return self.users[idx], self.items[idx], self.ratings[idx], features_idx, image_tensor, audio_tensor