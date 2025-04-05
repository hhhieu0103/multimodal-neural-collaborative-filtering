import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(
            self,
            df_interaction: pd.DataFrame,
            time_feature: str | None = None,
            df_metadata: pd.DataFrame = None,
            metadata_features: list | None = None,
    ):
        """
        Dataset for NCF model with support for time features and metadata.

        Args:
            df_interaction: Interaction dataframe with user_idx, item_idx, and rating columns
            time_feature: Name of the time feature column (None to disable)
            df_metadata: Item metadata dataframe (indexed by item_idx)
            metadata_features: List of metadata feature columns to use
        """
        users = df_interaction.iloc[:, 0].values
        items = df_interaction.iloc[:, 1].values
        ratings = df_interaction.iloc[:, 2].values

        # Ensure indices are properly cast as integers
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        # Time feature handling
        self.use_time = time_feature is not None
        if self.use_time:
            timestamps = df_interaction[time_feature].values
            self.timestamps = torch.tensor(timestamps, dtype=torch.float32)

        # Metadata feature handling
        self.use_metadata = df_metadata is not None and metadata_features is not None
        if self.use_metadata:
            self.feature_dims = []
            self.metadata = []

            # Process metadata if provided
            if df_metadata is not None and metadata_features:
                # Ensure metadata is indexed by item_idx
                if df_metadata.index.name != 'item_id' and 'item_id' in df_metadata.columns:
                    df_metadata = df_metadata.set_index('item_id')

                # Check if all items in interactions have metadata
                interaction_items = set(items)
                metadata_items = set(df_metadata.index)

                # Report stats on item overlap
                items_with_metadata = interaction_items.intersection(metadata_items)
                items_without_metadata = interaction_items - metadata_items

                print(f"Items in interactions: {len(interaction_items)}")
                print(f"Items in metadata: {len(metadata_items)}")
                print(f"Items in interactions with metadata: {len(items_with_metadata)}")
                print(f"Items in interactions WITHOUT metadata: {len(items_without_metadata)}")

                if items_without_metadata:
                    print("Warning: Some items in interactions don't have metadata")

                # Calculate feature dimensions
                for feature in metadata_features:
                    if feature not in df_metadata.columns:
                        print(f"Warning: Feature {feature} not found in metadata DataFrame")
                        continue

                    # Get a sample value to determine dimension
                    feature_samples = df_metadata[feature].dropna()
                    if len(feature_samples) == 0:
                        print(f"Warning: No non-null values found for feature {feature}")
                        continue

                    sample = feature_samples.iloc[0]

                    if isinstance(sample, list):
                        self.feature_dims.append(len(sample))
                    else:
                        self.feature_dims.append(1)

                # Create metadata tensors with robust handling for missing items
                for feature, dim in zip(metadata_features, self.feature_dims):
                    if feature not in df_metadata.columns:
                        continue

                    tensor = self.__create_metadata_tensors(
                        items=items,
                        feature=feature,
                        df_metadata=df_metadata,
                        feature_dim=dim
                    )
                    if tensor is not None:
                        self.metadata.append(tensor)

    @staticmethod
    def __create_metadata_tensors(items, feature, df_metadata, feature_dim):
        """
        Create tensor representations of metadata features.

        Args:
            items: Array of item indices from interactions
            feature: Name of the metadata feature
            df_metadata: DataFrame with metadata (indexed by item_idx)
            feature_dim: Dimension of the feature

        Returns:
            Tensor of shape [len(items), feature_dim]
        """
        # Create a matrix to store feature values
        matrix = np.zeros((len(items), feature_dim), dtype=np.float32)

        # Find which items exist in the metadata dataframe
        valid_items = []
        valid_indices = []

        for i, item in enumerate(items):
            if item in df_metadata.index:
                valid_items.append(item)
                valid_indices.append(i)

        valid_items = np.array(valid_items)
        valid_indices = np.array(valid_indices)

        if len(valid_items) == 0:
            print(f"Warning: No items matched in metadata for feature {feature}")
            return torch.tensor(matrix, dtype=torch.float32)

        try:
            if feature_dim == 1:
                # For numerical features
                # Get values and reshape to column vector
                values = df_metadata.loc[valid_items, feature].values

                # Handle case where values might be lists but should be scalar
                processed_values = []
                for v in values:
                    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                        processed_values.append(v[0])
                    else:
                        processed_values.append(v)

                # Reshape to column vector
                processed_values = np.array(processed_values).reshape(-1, 1)
                matrix[valid_indices] = processed_values

            else:
                # For categorical features (one-hot or multi-hot encoded)
                values = df_metadata.loc[valid_items, feature].values

                for i, idx in enumerate(valid_indices):
                    val = values[i]

                    if val is None or (not isinstance(val, (list, np.ndarray)) and pd.isna(val)):
                        # Handle null values with zeros
                        continue

                    if isinstance(val, (list, np.ndarray)):
                        # Check if dimensions match
                        if len(val) == feature_dim:
                            matrix[idx] = val
                        else:
                            print(
                                f"Warning: Feature {feature} has inconsistent dimension: expected {feature_dim}, got {len(val)}")
                    else:
                        # Handle scalar values for categorical features
                        print(f"Warning: Expected list for feature {feature}, got scalar value: {val}")

        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
            return torch.tensor(matrix, dtype=torch.float32)

        return torch.tensor(matrix, dtype=torch.float32)

    def get_feature_dims(self):
        """Get the dimensions of metadata features"""
        if self.use_metadata:
            return self.feature_dims
        return None

    def __len__(self):
        """Get the number of interactions in the dataset"""
        return self.users.size(0)

    def __getitem__(self, idx):
        """Get a single interaction with its features"""
        if self.use_time and self.use_metadata:
            metadata_idx = [feature[idx] for feature in self.metadata] if self.metadata else []
            return self.users[idx], self.items[idx], self.ratings[idx], self.timestamps[idx], metadata_idx

        elif self.use_time and not self.use_metadata:
            return self.users[idx], self.items[idx], self.ratings[idx], self.timestamps[idx]

        elif not self.use_time and self.use_metadata:
            metadata_idx = [feature[idx] for feature in self.metadata] if self.metadata else []
            return self.users[idx], self.items[idx], self.ratings[idx], metadata_idx

        elif not self.use_time and not self.use_metadata:
            return self.users[idx], self.items[idx], self.ratings[idx]