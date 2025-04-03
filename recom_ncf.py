import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF
import numpy as np
import pandas as pd
import gc
from time import time

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, positive_predictions, negative_predictions):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_predictions - negative_predictions)))
        return loss

class NCFRecommender:
    def __init__(
        self,
        unique_users,
        unique_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_time_dim=None,
        mlp_metadata_feature_dims=None,
        mlp_metadata_embedding_dims=None,
        num_mlp_layers=4,
        layers_ratio=2,
        learning_rate=0.001,
        epochs=20,
        dropout=0.0,
        weight_decay=0.0,
        loss_fn='bce',
        optimizer='sgd',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        
        self.val_losses = None
        self.train_losses = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.early_stopping_patience = min(3, round(epochs / 10))
        self.mlp_metadata_embedding_dims = mlp_metadata_embedding_dims
        
        self.unique_users = torch.tensor(unique_users)
        self.unique_items = torch.tensor(unique_items)

        self.use_time = mlp_time_dim is not None
        self.use_metadata = mlp_metadata_feature_dims is not None and mlp_metadata_embedding_dims is not None
        
        self.model = NCF(
            num_users=len(unique_users),
            num_items=len(unique_items),
            factors=factors,
            mlp_user_item_dim=mlp_user_item_dim,
            mlp_time_dim=mlp_time_dim,
            mlp_metadata_feature_dims=mlp_metadata_feature_dims,
            mlp_metadata_embedding_dims=mlp_metadata_embedding_dims,
            num_mlp_layers=num_mlp_layers,
            layers_ratio=layers_ratio,
            dropout=dropout
        ).to(self.device)
        
    def get_loss_function(self):
        if self.loss_fn == 'bce':
            return nn.BCELoss()
        elif self.loss_fn == 'mse':
            return nn.MSELoss()
        elif self.loss_fn == 'bpr':
            return BPRLoss()
        else:
            raise ValueError(f'Unsupported loss function: {self.loss_fn}')
    
    def get_optimizer(self):
        if self.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')
    
    def fit(self, train_data: DataLoader, val_data: DataLoader):
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        loss_fn = self.get_loss_function()
        optimizer = self.get_optimizer()
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train(train_data, loss_fn, optimizer)
            train_losses.append(train_loss)
            
            val_loss = self.validate(val_data, loss_fn)
            val_losses.append(val_loss)
            
            print(f'Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            print('='*50)
            
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        self.train_losses = train_losses
        self.val_losses = val_losses
        print("Training completed!")
    
    def train(self, train_data, loss_fn, optimizer):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for data_batch in train_data:
            users, items, ratings, timestamps, metadata = self._move_data_to_device(data_batch)

            optimizer.zero_grad()

            predictions = self.model(users, items, timestamps, metadata)
            loss = loss_fn(predictions, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, val_data, loss_fn):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data_batch in val_data:
                users, items, ratings, timestamps, metadata = self._move_data_to_device(data_batch)
                
                predictions = self.model(users, items, timestamps, metadata)
                loss = loss_fn(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches

    def _move_data_to_device(self, data_batch):
        users, items, ratings, timestamps, metadata = None, None, None, None, None

        if self.use_time and self.use_metadata:
            users, items, ratings, timestamps, metadata = data_batch
        elif self.use_time and not self.use_metadata:
            users, items, ratings, timestamps = data_batch
        elif not self.use_time and self.use_metadata:
            users, items, ratings, metadata = data_batch
        else:
            users, items, ratings = data_batch

        users = users.to(self.device)
        items = items.to(self.device)
        ratings = ratings.to(self.device)

        if self.use_time and timestamps is not None:
            timestamps = timestamps.to(self.device)
        else:
            timestamps = None

        metadata_on_device = None
        if self.use_metadata and metadata is not None:
            metadata_on_device = []
            for feature in metadata:
                metadata_on_device.append(feature.to(self.device))

        return users, items, ratings, timestamps, metadata_on_device
    
    def predict(
        self,
        users: torch.Tensor | np.ndarray,
        items: torch.Tensor | np.ndarray,
        timestamps: torch.Tensor | np.ndarray,
        metadata
    ):
        self.model.eval()
        
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long)

        if not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long)

        users = users.to(self.device)
        items = items.to(self.device)

        if self.use_time and not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
            timestamps = timestamps.to(self.device)

        metadata_tensors = None
        if self.use_metadata and metadata is not None:
            metadata_tensors = []
            for feature in metadata:
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature, dtype=torch.float32)
                metadata_tensors.append(feature.to(self.device))
        
        with torch.no_grad():
            predictions = self.model(users, items, timestamps, metadata_tensors)
            
        return predictions.cpu()

    def batch_predict_for_users(
            self,
            users: torch.Tensor | np.ndarray,
            timestamps: torch.Tensor | np.ndarray | None = None,
            df_metadata: pd.DataFrame | None = None,
            metadata_features=None,
            items: torch.Tensor | np.ndarray | None = None,
            k=10,
            user_batch_size=1024,
            item_batch_size=8096,
    ):
        """
        Generate predictions for all items for each user, with dynamic batch size adjustment
        to optimize GPU memory usage.

        Args:
            users: List or array of user IDs
            timestamps: Timestamps for each user (optional, used if model supports time)
            df_metadata: DataFrame containing metadata features for items
            metadata_features: List of metadata feature column names to use
            items: Optional tensor of item IDs. If None, all items are used
            k: Number of top items to retrieve per user
            user_batch_size: Initial number of users to process in each batch
            item_batch_size: Initial number of items to process in each batch

        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        # Step 1: Setup and initialization
        self.model.eval()
        predictions = {}
        start_time = time()

        # Process input data to standard formats
        users_np = self._convert_to_numpy(users)
        timestamps_np = self._process_timestamps(timestamps)
        items = self._setup_items_tensor(items)

        # Get dimensions
        num_users = len(users_np)
        num_items = len(items)

        # Initialize batch sizes (will be adjusted dynamically)
        current_user_batch_size = user_batch_size
        current_item_batch_size = item_batch_size

        print(f"Processing predictions for {num_users} users and {num_items} items")

        # Step 2: Precompute metadata features if needed
        metadata_cache = self._precompute_metadata(df_metadata, metadata_features, items)

        # Step 3: Process all users in batches
        with torch.no_grad():
            i = 0
            while i < num_users:
                # Get the next batch of users
                end_idx = min(i + current_user_batch_size, num_users)
                user_batch = users_np[i:end_idx]
                batch_size = len(user_batch)

                # Log progress
                print(f'Processing {i + 1} of {num_users} users... ({i / num_users:.2%})')

                # Prepare user tensors
                users_tensor = torch.tensor(user_batch, dtype=torch.long, device=self.device)

                # Prepare timestamp tensors if needed
                timestamps_tensor = None
                if self.use_time and timestamps_np is not None:
                    batch_timestamps = timestamps_np[i:end_idx]
                    timestamps_tensor = torch.tensor(batch_timestamps, dtype=torch.float32, device=self.device)

                # Initialize tensor to store all scores for this batch
                all_scores = torch.zeros((batch_size, num_items), device=self.device)

                # Process items in batches for the current user batch
                j = 0
                while j < num_items:
                    try:
                        # Get the current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        item_batch = items[j:end_item_idx]

                        # Process the batch of items for all users
                        batch_scores = self._process_item_batch(
                            users_tensor,
                            item_batch,
                            timestamps_tensor,
                            metadata_cache,
                            metadata_features
                        )

                        # Store scores in the all_scores tensor
                        all_scores[:, j:end_item_idx] = batch_scores

                        # Move to next item batch
                        j = end_item_idx

                    except RuntimeError as e:
                        # Handle OOM errors by reducing batch size and retrying
                        if "CUDA out of memory" in str(e):
                            current_item_batch_size = self._handle_oom_error(current_item_batch_size)
                            # Don't advance the index - retry with smaller batch
                            continue
                        else:
                            # Re-raise other errors
                            raise e

                # Extract top-k items for each user in the batch
                self._extract_top_k_items(user_batch, all_scores, k, predictions)

                # print('After item loop', self._get_gpu_memory_usage())
                current_user_batch_size, current_item_batch_size = self._adjust_batch_size(current_user_batch_size, current_item_batch_size)

                # Clean up batch memory
                del all_scores, users_tensor, timestamps_tensor
                torch.cuda.empty_cache()
                gc.collect()

                # Move to next user batch
                i = end_idx

        total_time = time() - start_time
        print(f"Prediction completed in {total_time:.2f} seconds")
        return predictions

    # Helper methods for batch_predict_for_users

    def _convert_to_numpy(self, tensor_or_array):
        """Convert tensor or array to numpy array"""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.cpu().numpy()
        else:
            return np.array(tensor_or_array)

    def _process_timestamps(self, timestamps):
        """Process timestamps to numpy array if they exist"""
        if not self.use_time or timestamps is None:
            return None

        if isinstance(timestamps, torch.Tensor):
            return timestamps.cpu().numpy()
        else:
            return np.array(timestamps) if timestamps is not None else None

    def _setup_items_tensor(self, items):
        """Setup the items tensor"""
        if items is None:
            return torch.arange(len(self.unique_items), device=self.device)
        elif not isinstance(items, torch.Tensor):
            return torch.tensor(items, dtype=torch.long, device=self.device)
        return items

    def _get_user_batches(self, users_np, batch_size):
        """Generate batches of users"""
        for i in range(0, len(users_np), batch_size):
            yield users_np[i:i + batch_size]

    def _get_item_batches(self, items, batch_size):
        """Generate batches of items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _prepare_user_batch_tensors(self, user_batch, timestamps_np):
        """Prepare tensor representations for a user batch"""
        users_tensor = torch.tensor(user_batch, dtype=torch.long, device=self.device)

        timestamps_tensor = None
        if self.use_time and timestamps_np is not None:
            batch_indices = np.searchsorted(np.arange(len(timestamps_np)), user_batch)
            valid_indices = batch_indices < len(timestamps_np)

            if np.all(valid_indices):
                batch_timestamps = timestamps_np[batch_indices]
                timestamps_tensor = torch.tensor(batch_timestamps, dtype=torch.float32, device=self.device)

        return users_tensor, timestamps_tensor

    def _precompute_metadata(self, df_metadata, metadata_features, items):
        """Precompute metadata features for faster access during prediction"""
        if not self.use_metadata or df_metadata is None or metadata_features is None:
            return None

        start = time()
        print("Precomputing metadata features...")

        # Ensure df_metadata is indexed properly
        if 'item_idx' in df_metadata.columns and df_metadata.index.name != 'item_idx':
            df_metadata = df_metadata.set_index('item_idx')

        # Initialize the metadata cache
        metadata_cache = {}
        items_np = items.cpu().numpy()

        # Process each feature
        for feature in metadata_features:
            if feature not in df_metadata.columns:
                print(f"Warning: Feature '{feature}' not found in metadata")
                continue

            # Sample the first non-null value to determine feature dimension
            sample_value = None
            for idx in df_metadata.index:
                if idx in df_metadata.index:
                    val = df_metadata.loc[idx, feature]
                    if val is not None:
                        sample_value = val
                        break

            if sample_value is None:
                print(f"Warning: No valid values found for feature '{feature}'")
                continue

            # Determine feature dimension
            feature_dim = len(sample_value) if isinstance(sample_value, (list, np.ndarray)) else 1
            is_high_dim = feature_dim > 500  # Special handling for high-dimensional features

            # Initialize feature cache
            metadata_cache[feature] = {
                'data': {},
                'dim': feature_dim,
                'is_high_dim': is_high_dim
            }

            # Cache feature values for all items
            for item_id in items_np:
                if item_id in df_metadata.index:
                    val = df_metadata.loc[item_id, feature]

                    if val is not None:
                        if is_high_dim and isinstance(val, list):
                            # For high-dim features, store only non-zero indices to save memory
                            non_zero_indices = [i for i, v in enumerate(val) if v != 0]
                            metadata_cache[feature]['data'][item_id] = non_zero_indices
                        else:
                            metadata_cache[feature]['data'][item_id] = val
                    else:
                        # Handle missing values
                        metadata_cache[feature]['data'][item_id] = (
                            [0.0] * feature_dim if feature_dim > 1 else 0.0
                        )
                else:
                    # Handle items not in metadata
                    metadata_cache[feature]['data'][item_id] = (
                        [0.0] * feature_dim if feature_dim > 1 else 0.0
                    )

        print(f"Metadata precomputation completed in {time() - start:.2f} seconds")
        return metadata_cache

    def _process_item_batch(self, users_tensor, item_batch, timestamps_tensor, metadata_cache, metadata_features):
        """Process a batch of items for all users in the current batch"""
        batch_size = users_tensor.size(0)  # Number of users
        item_batch_size = len(item_batch)  # Number of items

        # Create input matrices for the model
        users_matrix = users_tensor.unsqueeze(1).expand(-1, item_batch_size).reshape(-1)
        items_matrix = item_batch.unsqueeze(0).expand(batch_size, -1).reshape(-1)

        # Process timestamps if available
        timestamps_matrix = None
        if self.use_time and timestamps_tensor is not None:
            timestamps_matrix = timestamps_tensor.unsqueeze(1).expand(-1, item_batch_size).reshape(-1)

        # Process metadata if available
        metadata_matrices = None
        if self.use_metadata and metadata_cache:
            metadata_matrices = self._create_metadata_matrices(
                item_batch, batch_size, metadata_cache, metadata_features
            )

        # Make predictions
        batch_predictions = self.model(users_matrix, items_matrix, timestamps_matrix, metadata_matrices)

        # Reshape predictions to batch_size x item_batch_size
        return batch_predictions.view(batch_size, item_batch_size)

    def _create_metadata_matrices(self, item_batch, batch_size, metadata_cache, metadata_features):
        """Create metadata matrices for a batch of items"""
        batch_items_np = item_batch.cpu().numpy()
        metadata_matrices = []

        for feature in metadata_features:
            if feature not in metadata_cache:
                continue

            feature_info = metadata_cache[feature]
            feature_data = feature_info['data']
            feature_dim = feature_info['dim']
            is_high_dim = feature_info['is_high_dim']

            if is_high_dim:
                # For high-dimensional sparse features
                batch_tensor = torch.zeros(
                    (len(batch_items_np), feature_dim),
                    dtype=torch.float32,
                    device=self.device
                )

                # Set the non-zero values
                for i, item_id in enumerate(batch_items_np):
                    if item_id in feature_data:
                        indices = feature_data[item_id]
                        if indices:  # If there are non-zero indices
                            batch_tensor[i, indices] = 1.0
            else:
                # For regular features
                if feature_dim > 1:
                    # Multi-dimensional features
                    values = [feature_data.get(item_id, [0.0] * feature_dim) for item_id in batch_items_np]
                    batch_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                else:
                    # Single-dimensional features
                    values = [feature_data.get(item_id, 0.0) for item_id in batch_items_np]
                    batch_tensor = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(1)

            # Expand the tensor for all users
            expanded_tensor = batch_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            expanded_tensor = expanded_tensor.reshape(-1, expanded_tensor.size(-1))
            metadata_matrices.append(expanded_tensor)

        return metadata_matrices

    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

            # Estimate total GPU memory (adjust based on your GPU)
            total_gpu_memory = 8.0  # RTX 4060 Mobile has around 8GB

            gpu_memory_usage = reserved / total_gpu_memory
            return gpu_memory_usage
        return 0

    def _adjust_batch_size(self, current_user_batch_size, current_item_batch_size, min_user_batch_size=64, min_item_batch_size=128):
        """Adjust item batch size based on current GPU memory usage"""
        gpu_memory_usage = self._get_gpu_memory_usage()

        if gpu_memory_usage > 0.95:

            if current_item_batch_size > min_item_batch_size:
                new_item_batch_size = max(min_item_batch_size, current_item_batch_size // 2)
                if new_item_batch_size != current_item_batch_size:
                    print('Memory usage:', gpu_memory_usage ,'. Reduced item batch size from', current_item_batch_size, 'to', new_item_batch_size)
                    current_item_batch_size = new_item_batch_size
            elif current_user_batch_size > min_user_batch_size:
                new_user_batch_size = max(min_user_batch_size, current_user_batch_size // 2)
                if new_user_batch_size != current_user_batch_size:
                    print('Memory usage:', gpu_memory_usage ,'. Reduced user batch size from', current_user_batch_size, 'to', new_user_batch_size)
                    current_user_batch_size = new_user_batch_size
            else:
                current_user_batch_size = min_user_batch_size
                current_item_batch_size = min_item_batch_size

        elif gpu_memory_usage < 0.8:

            increasing_rate = 0.95 / gpu_memory_usage

            if self.use_time:
                increasing_rate = increasing_rate * 0.85
            if self.use_metadata:
                num_metadata_features = len(self.mlp_metadata_embedding_dims)
                increasing_rate = increasing_rate * max(0.1, (0.95 - 0.1 * num_metadata_features))

            new_user_batch_size = round(current_user_batch_size * increasing_rate)
            new_item_batch_size = round(current_item_batch_size * increasing_rate)

            if new_user_batch_size != current_user_batch_size:
                print('Memory usage:', gpu_memory_usage ,'. Increased user batch size from', current_user_batch_size, 'to', new_user_batch_size)
                current_user_batch_size = new_user_batch_size

            if new_item_batch_size != current_item_batch_size:
                print('Memory usage:', gpu_memory_usage ,'. Increased item batch size from', current_item_batch_size, 'to', new_item_batch_size)
                current_item_batch_size = new_item_batch_size

        return current_user_batch_size, current_item_batch_size

    def _handle_oom_error(self, current_item_batch_size):
        """Handle out of memory error by reducing batch size"""
        reduced_batch_size = max(128, current_item_batch_size // 4)

        print(f"CUDA OOM error! Reduced item batch size to {reduced_batch_size}")

        # Clear memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return reduced_batch_size

    def _extract_top_k_items(self, user_batch, all_scores, k, predictions):
        """Extract top-k items for each user from scores"""
        user_scores = all_scores.cpu().numpy()

        for i, user_id in enumerate(user_batch):
            # Find top-k items efficiently
            scores = user_scores[i]
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

            # Store results
            predictions[user_id] = top_k_indices.tolist()
