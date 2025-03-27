import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, df_metadata: pd.DataFrame, metadata_features=[], time_feature='timestamp'):
        users = df.iloc[:, 0].values
        items = df.iloc[:, 1].values
        ratings = df.iloc[:, 2].values
        timestamps = df[time_feature].values
        
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        
        # Vectorized metadata handling
        if metadata_features:
            
            # Pre-process metadata DataFrame
            df_metadata = df_metadata.set_index('item_idx')
            
            # Prepare a matrix for all metadata features
            metadata_matrix = np.zeros((len(items), len(metadata_features)), dtype=np.float32)
            
            # Create mask for valid items (those present in metadata)
            valid_items_mask = np.isin(items, df_metadata.index)
            valid_indices = np.where(valid_items_mask)[0]
            valid_items = items[valid_indices]
            
            # Fetch all required metadata at once for valid items
            if len(valid_items) > 0:
                # Get metadata for valid items
                features_data = df_metadata.loc[valid_items, metadata_features].values
                
                # Place the data in the correct positions in our matrix
                metadata_matrix[valid_indices] = features_data
            
            # Convert to list of torch tensors (one per feature)
            self.metadata = [torch.tensor(metadata_matrix[:, i], dtype=torch.float32) 
                             for i in range(len(metadata_features))]
        else:
            self.metadata = []

    def __len__(self):
        return self.users.size(0)

    def __getitem__(self, idx):
        metadata_idx = [feature[idx] for feature in self.metadata]
        return self.users[idx], self.items[idx], self.ratings[idx], self.timestamps[idx], metadata_idx