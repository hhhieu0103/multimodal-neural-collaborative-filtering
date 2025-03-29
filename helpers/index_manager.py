import pandas as pd
import numpy as np

class IndexManager:
    """
    Manages the mapping between original IDs and consecutive indices
    for both users and items in a recommender system.
    """

    def __init__(self):
        self.user_id_to_idx = {}
        self.user_idx_to_id = {}
        self.item_id_to_idx = {}
        self.item_idx_to_id = {}

        # Counter for newly added items
        self.next_user_idx = 0
        self.next_item_idx = 0

    def fit(self, df_interaction, user_col='user_id', item_col='item_id', df_metadata=None):
        """
        Create mappings from the interaction and metadata dataframes.

        Args:
            df_interaction: DataFrame with user-item interactions
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            df_metadata: Optional metadata DataFrame with additional items

        Returns:
            self for method chaining
        """
        # Process user IDs from interactions
        unique_users = df_interaction[user_col].unique()
        for user_id in unique_users:
            self.user_id_to_idx[user_id] = self.next_user_idx
            self.user_idx_to_id[self.next_user_idx] = user_id
            self.next_user_idx += 1

        # Process item IDs from interactions
        unique_items = set(df_interaction[item_col].unique())

        # Add items from metadata if provided
        if df_metadata is not None:
            metadata_id_col = item_col
            if df_metadata.index.name == item_col:
                metadata_items = set(df_metadata.index)
            elif item_col in df_metadata.columns:
                metadata_items = set(df_metadata[item_col])
            else:
                raise ValueError(f"Item column '{item_col}' not found in metadata")

            # Combine items from both sources
            unique_items = unique_items.union(metadata_items)

        # Create consecutive indices for all items
        for item_id in sorted(unique_items):
            self.item_id_to_idx[item_id] = self.next_item_idx
            self.item_idx_to_id[self.next_item_idx] = item_id
            self.next_item_idx += 1

        print(f"Indexed {len(self.user_id_to_idx)} users and {len(self.item_id_to_idx)} items")
        print(f"User index range: 0-{self.next_user_idx-1}")
        print(f"Item index range: 0-{self.next_item_idx-1}")

        return self

    def transform_interactions(self, df_interaction, user_col='user_id', item_col='item_id',
                              inplace=False, add_missing=True):
        """
        Transform user and item IDs to consecutive indices in the interaction DataFrame.

        Args:
            df_interaction: DataFrame with user-item interactions
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            inplace: If True, modify the DataFrame in-place
            add_missing: If True, add missing IDs to the mappings

        Returns:
            DataFrame with transformed indices
        """
        df = df_interaction if inplace else df_interaction.copy()

        # Handle users
        if add_missing:
            # Add any new users to the mapping
            new_users = set(df[user_col]) - set(self.user_id_to_idx.keys())
            for user_id in new_users:
                self.user_id_to_idx[user_id] = self.next_user_idx
                self.user_idx_to_id[self.next_user_idx] = user_id
                self.next_user_idx += 1

            # Map user IDs to indices
            df[user_col] = df[user_col].map(self.user_id_to_idx)
        else:
            # Only map existing users, drop rows with unknown users
            mask = df[user_col].isin(self.user_id_to_idx)
            if not mask.all():
                n_dropped = (~mask).sum()
                print(f"Dropped {n_dropped} rows with unknown users")
                df = df[mask]
            df[user_col] = df[user_col].map(self.user_id_to_idx)

        # Handle items
        if add_missing:
            # Add any new items to the mapping
            new_items = set(df[item_col]) - set(self.item_id_to_idx.keys())
            for item_id in new_items:
                self.item_id_to_idx[item_id] = self.next_item_idx
                self.item_idx_to_id[self.next_item_idx] = item_id
                self.next_item_idx += 1

            # Map item IDs to indices
            df[item_col] = df[item_col].map(self.item_id_to_idx)
        else:
            # Only map existing items, drop rows with unknown items
            mask = df[item_col].isin(self.item_id_to_idx)
            if not mask.all():
                n_dropped = (~mask).sum()
                print(f"Dropped {n_dropped} rows with unknown items")
                df = df[mask]
            df[item_col] = df[item_col].map(self.item_id_to_idx)

        return df

    def transform_metadata(self, df_metadata, item_col='item_id', inplace=False,
                          add_missing=True, set_index=True):
        """
        Transform item IDs to consecutive indices in the metadata DataFrame.

        Args:
            df_metadata: Metadata DataFrame
            item_col: Column name for item IDs
            inplace: If True, modify the DataFrame in-place
            add_missing: If True, add missing IDs to the mappings
            set_index: If True, set the transformed item column as index

        Returns:
            DataFrame with transformed indices
        """
        df = df_metadata if inplace else df_metadata.copy()

        # Extract the item IDs depending on whether they're in the index or a column
        if df.index.name == item_col:
            item_ids = df.index
            # Reset index to make it a regular column for transformation
            df = df.reset_index()
        elif item_col in df.columns:
            item_ids = df[item_col]
        else:
            raise ValueError(f"Item column '{item_col}' not found in metadata")

        # Handle items
        if add_missing:
            # Add any new items to the mapping
            new_items = set(item_ids) - set(self.item_id_to_idx.keys())
            for item_id in new_items:
                self.item_id_to_idx[item_id] = self.next_item_idx
                self.item_idx_to_id[self.next_item_idx] = item_id
                self.next_item_idx += 1

            # Map item IDs to indices
            df[item_col] = df[item_col].map(self.item_id_to_idx)
        else:
            # Only keep items we know about
            mask = df[item_col].isin(self.item_id_to_idx)
            if not mask.all():
                n_dropped = (~mask).sum()
                print(f"Dropped {n_dropped} metadata entries with unknown items")
                df = df[mask]
            df[item_col] = df[item_col].map(self.item_id_to_idx)

        # Set the transformed column as index if requested
        if set_index:
            df = df.set_index(item_col)

        return df

    def get_indexed_users(self):
        """Return a list of all user indices in consecutive order"""
        return list(range(len(self.user_id_to_idx)))

    def get_indexed_items(self):
        """Return a list of all item indices in consecutive order"""
        return list(range(len(self.item_id_to_idx)))

    def predicted_id_to_idx(self, users, items=None, timestamps=None):
        """
        Convert user and item IDs to indices for prediction.

        Args:
            users: List or array of user IDs
            items: Optional list or array of item IDs
            timestamps: Optional list or array of timestamps

        Returns:
            Tuple of (user_indices, item_indices, timestamps)
        """
        # Convert users to indices
        user_indices = np.array([self.user_id_to_idx.get(u, -1) for u in users])

        # Check if all users were found
        if (user_indices == -1).any():
            missing_count = (user_indices == -1).sum()
            print(f"Warning: {missing_count} users not found in the index mapping")
            # Filter out missing users
            valid_mask = user_indices != -1
            user_indices = user_indices[valid_mask]
            if timestamps is not None:
                timestamps = timestamps[valid_mask]

        # Convert items to indices if provided
        item_indices = None
        if items is not None:
            item_indices = np.array([self.item_id_to_idx.get(i, -1) for i in items])

            # Check if all items were found
            if (item_indices == -1).any():
                missing_count = (item_indices == -1).sum()
                print(f"Warning: {missing_count} items not found in the index mapping")
                # Filter out missing items
                item_indices = item_indices[item_indices != -1]

        return user_indices, item_indices, timestamps

    def convert_predictions_to_ids(self, user_indices, item_prediction_indices):
        """
        Convert prediction results from indices back to original IDs.

        Args:
            user_indices: List or array of user indices
            item_prediction_indices: Dictionary mapping user indices to lists of recommended item indices

        Returns:
            Dictionary mapping original user IDs to lists of recommended item IDs
        """
        result = {}

        for user_idx in user_indices:
            if user_idx in item_prediction_indices:
                # Get the original user ID
                user_id = self.user_idx_to_id.get(user_idx)

                if user_id is not None:
                    # Convert item indices to original IDs
                    item_indices = item_prediction_indices[user_idx]
                    item_ids = [self.item_idx_to_id.get(idx) for idx in item_indices]

                    # Remove any None values (could happen if index is out of range)
                    item_ids = [item_id for item_id in item_ids if item_id is not None]

                    result[user_id] = item_ids

        return result

    def save(self, filepath):
        """Save the index mappings to a file"""
        import json

        # Convert numeric keys to strings for JSON serialization
        data = {
            'user_id_to_idx': self.user_id_to_idx,
            'user_idx_to_id': {str(k): v for k, v in self.user_idx_to_id.items()},
            'item_id_to_idx': self.item_id_to_idx,
            'item_idx_to_id': {str(k): v for k, v in self.item_idx_to_id.items()},
            'next_user_idx': self.next_user_idx,
            'next_item_idx': self.next_item_idx
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Load index mappings from a file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        manager = cls()
        manager.user_id_to_idx = data['user_id_to_idx']
        manager.user_idx_to_id = {int(k): v for k, v in data['user_idx_to_id'].items()}
        manager.item_id_to_idx = data['item_id_to_idx']
        manager.item_idx_to_id = {int(k): v for k, v in data['item_idx_to_id'].items()}
        manager.next_user_idx = data['next_user_idx']
        manager.next_item_idx = data['next_item_idx']

        return manager


# Example usage:
"""
# Create and fit the index manager
index_manager = IndexManager().fit(
    df_interaction=train_data,
    df_metadata=metadata
)

# Transform your datasets to use consecutive indices
train_data_indexed = index_manager.transform_interactions(train_data)
val_data_indexed = index_manager.transform_interactions(val_data)
test_data_indexed = index_manager.transform_interactions(test_data)
metadata_indexed = index_manager.transform_metadata(metadata)

# Get consecutive indices for model initialization
unique_users = index_manager.get_indexed_users()
unique_items = index_manager.get_indexed_items()

# Initialize the model with these indices
recommender = NCFRecommender(
    unique_users=unique_users,
    unique_items=unique_items,
    # other parameters...
)

# Train and evaluate with the indexed data
# ...

# When predicting, convert original IDs to indices
user_indices, _, timestamps = index_manager.predict_id_to_idx(
    users=test_users, 
    timestamps=test_timestamps
)

# Make predictions
predictions = recommender.batch_predict_for_users(
    users=user_indices,
    timestamps=timestamps,
    df_metadata=metadata_indexed,
    metadata_features=metadata_features,
    k=10
)

# Convert predictions back to original IDs
predictions_with_ids = index_manager.convert_predictions_to_ids(
    user_indices=user_indices,
    item_prediction_indices=predictions
)
"""