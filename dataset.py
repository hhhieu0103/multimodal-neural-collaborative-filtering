import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, df_metadata: pd.DataFrame = None, metadata_features=[],
                 time_feature='timestamp'):
        users = df.iloc[:, 0].values
        items = df.iloc[:, 1].values
        ratings = df.iloc[:, 2].values
        timestamps = df[time_feature].values

        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)

        self.feature_dims = []
        self.metadata = []

        # Process metadata if provided
        if df_metadata is not None and metadata_features:
            # Create a mapping of item_idx to its row index for faster lookups
            if 'item_idx' in df_metadata.columns:
                df_metadata = df_metadata.set_index('item_idx')

            # Calculate feature dimensions
            for feature in metadata_features:
                if feature not in df_metadata.columns:
                    print(f"Warning: Feature {feature} not found in metadata DataFrame")
                    continue

                # Get a sample value to determine dimension
                non_null_indices = df_metadata[feature].dropna().index
                if len(non_null_indices) == 0:
                    print(f"Warning: No non-null values found for feature {feature}")
                    continue

                sample = df_metadata[feature].loc[non_null_indices[0]]

                if isinstance(sample, list):
                    self.feature_dims.append(len(sample))
                else:
                    self.feature_dims.append(1)

            # Create metadata tensors
            for feature, dim in zip(metadata_features, self.feature_dims):
                if feature not in df_metadata.columns:
                    continue

                tensor = self.__create_metadata_tensors(items=items, feature=feature, df_metadata=df_metadata,
                                                        feature_dim=dim)
                if tensor is not None:
                    self.metadata.append(tensor)

    def __create_metadata_tensors(self, items, feature, df_metadata, feature_dim):
        # Create a matrix to store feature values
        matrix = np.zeros((len(items), feature_dim), dtype=np.float32)

        # Find which items are in the metadata dataframe
        valid_items_mask = np.isin(items, df_metadata.index)
        valid_indices = np.where(valid_items_mask)[0]
        valid_items = items[valid_indices]

        if len(valid_items) == 0:
            print(f"Warning: No items matched in metadata for feature {feature}")
            return None

        try:
            if feature_dim == 1:
                # For numerical features
                # Get values and reshape to column vector
                values = df_metadata.loc[valid_items, feature].values
                if isinstance(values[0], (list, np.ndarray)):
                    # Handle case where values are lists but should be scalar
                    values = np.array([v[0] if isinstance(v, (list, np.ndarray)) else v for v in values])
                matrix[valid_indices] = values.reshape(-1, 1)
            else:
                # For categorical features (one-hot or multi-hot encoded)
                # Convert lists to numpy arrays and stack them
                values = df_metadata.loc[valid_items, feature].values
                stacked_values = np.zeros((len(values), feature_dim), dtype=np.float32)

                for i, val in enumerate(values):
                    if val is not None:
                        if isinstance(val, (list, np.ndarray)) and len(val) == feature_dim:
                            stacked_values[i] = val
                        else:
                            print(f"Warning: Unexpected format for feature {feature}, item {valid_items[i]}: {val}")

                matrix[valid_indices] = stacked_values
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
            return None

        return torch.tensor(matrix, dtype=torch.float32)

    def get_feature_dims(self):
        return self.feature_dims

    def __len__(self):
        return self.users.size(0)

    def __getitem__(self, idx):
        metadata_idx = [feature[idx] for feature in self.metadata] if self.metadata else []
        return self.users[idx], self.items[idx], self.ratings[idx], self.timestamps[idx], metadata_idx