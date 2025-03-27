import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF
import numpy as np
import pandas as pd

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, positive_predictions, negative_predictions):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_predictions - negative_predictions)))
        return loss

class NCFRecommender():
    def __init__(
        self,
        unique_users,
        unique_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_time_dim=8,
        mlp_metadata_feature_dims=[],
        mlp_metadata_embedding_dims=[],
        num_mlp_layers=4,
        layers_ratio=2,
        learning_rate=0.001,
        epochs=20,
        dropout=0.0,
        weight_decay=0.0,
        loss_fn='bce',
        optimizer='sgd',
        early_stopping_patience=5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        self.unique_users = torch.tensor(unique_users)
        self.unique_items = torch.tensor(unique_items)
        
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
        if (self.loss_fn == 'bce'):
            return nn.BCELoss()
        elif (self.loss_fn == 'mse'):
            return nn.MSELoss()
        elif (self.loss_fn == 'bpr'):
            return BPRLoss()
        else:
            raise ValueError(f'Unsupported loss function: {self.loss_fn}')
    
    def get_optimizer(self):
        if (self.optimizer == 'sgd'):
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif (self.optimizer == 'adam'):
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif (self.optimizer == 'adagrad'):
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
        
        for users, items, ratings, timestamps, metadata in train_data:
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)
            timestamps = timestamps.to(self.device)
            metadata = [feature.to(self.device) for feature in metadata]
            
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
            for users, items, ratings, timestamps, metadata in val_data:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                timestamps = timestamps.to(self.device)
                metadata = [feature.to(self.device) for feature in metadata]
                
                predictions = self.model(users, items, timestamps, metadata)
                loss = loss_fn(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
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
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
            
        users = users.to(self.device)
        items = items.to(self.device)
        timestamps = timestamps.to(self.device)
        
        for feature in metadata:
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature, dtype=torch.float32)
            feature = feature.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(users, items, timestamps, metadata)
            
        return predictions.cpu()

    def batch_predict_for_users(
            self,
            users: torch.Tensor | np.ndarray,
            timestamps: torch.Tensor | np.ndarray,
            df_metadata: pd.DataFrame = None,
            metadata_features=[],
            items: torch.Tensor | np.ndarray | None = None,
            k=10,
            user_batch_size=1024,
            item_batch_size=4096,
            memory_threshold=0.95  # GPU memory utilization threshold (0-1)
    ):
        """
        Generate predictions for all items for each user with dynamic batch size adjustment
        based on GPU memory usage to avoid expensive shared memory operations

        Args:
            users: List or array of user IDs
            timestamps: Timestamps for each user
            df_metadata: DataFrame containing metadata features for items
            metadata_features: List of metadata feature column names to use
            items: Optional tensor of item IDs. If None, all items are used.
            k: Number of top items to retain per user
            user_batch_size: Initial number of users to process in each batch
            item_batch_size: Initial number of items to process in each batch
            memory_threshold: GPU memory utilization threshold (0-1) to trigger batch size reduction

        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        import gc
        from time import time

        self.model.eval()
        predictions = {}

        # Convert users to numpy for iteration if needed
        if isinstance(users, torch.Tensor):
            users_np = users.cpu().numpy()
        else:
            users_np = np.array(users)

        if isinstance(timestamps, torch.Tensor):
            timestamps_np = timestamps.cpu().numpy()
        else:
            timestamps_np = np.array(timestamps)

        # Setup item tensor on device
        if items is None:
            items = torch.arange(len(self.unique_items), device=self.device)
        elif not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long, device=self.device)

        num_users = len(users_np)
        num_items = len(items)

        # Current batch sizes (will be adjusted dynamically)
        current_user_batch_size = user_batch_size
        current_item_batch_size = item_batch_size

        # Track if we're using shared memory
        using_shared_memory = False

        print(f"Processing predictions for {num_users} users and {num_items} items")
        print(f"Initial batch sizes - Users: {current_user_batch_size}, Items: {current_item_batch_size}")

        # Precompute metadata features for all items
        metadata_cache = {}
        if df_metadata is not None and len(metadata_features) > 0:
            print("Precomputing metadata features...")

            # Ensure df_metadata is indexed by item_idx
            if 'item_idx' in df_metadata.columns:
                df_metadata = df_metadata.set_index('item_idx')

            # Create a lookup table for each feature
            for feature in metadata_features:
                if feature not in df_metadata.columns:
                    print(f"Warning: Feature {feature} not found in metadata")
                    continue

                metadata_cache[feature] = {}

                # Find a valid sample to determine feature dimension
                sample = None
                for idx in df_metadata.index:
                    val = df_metadata.loc[idx, feature]
                    if val is not None:
                        sample = val
                        break

                # Determine feature dimension
                feature_dim = len(sample) if isinstance(sample, list) else 1

                # Precompute and store all item features
                for item_id in items.cpu().numpy():
                    if item_id in df_metadata.index:
                        val = df_metadata.loc[item_id, feature]
                        if val is not None:
                            metadata_cache[feature][item_id] = val
                        else:
                            # Handle missing values
                            metadata_cache[feature][item_id] = [0.0] * feature_dim if feature_dim > 1 else 0.0
                    else:
                        # Handle items not in metadata
                        metadata_cache[feature][item_id] = [0.0] * feature_dim if feature_dim > 1 else 0.0

        # Function to check GPU memory usage
        def get_gpu_memory_usage():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB

                # Estimate total GPU memory - adjust based on your GPU
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                gpu_memory_usage = reserved / total_gpu_memory
                return reserved, allocated, gpu_memory_usage
            return 0, 0, 0

        # Function to adjust batch sizes based on memory usage
        def adjust_batch_sizes():
            nonlocal current_user_batch_size, current_item_batch_size, using_shared_memory

            reserved, allocated, gpu_memory_usage = get_gpu_memory_usage()

            # Print memory stats
            print(f"GPU Memory: Reserved={reserved:.2f}GB ({gpu_memory_usage * 100:.1f}%), Allocated={allocated:.2f}GB")
            print(f"Current batch sizes - Users: {current_user_batch_size}, Items: {current_item_batch_size}")

            # Determine if the GPU is close to running out of memory
            if gpu_memory_usage > memory_threshold and not using_shared_memory:
                # We're likely using shared memory
                using_shared_memory = True

                # Reduce batch sizes by 50%
                current_user_batch_size = max(32, current_user_batch_size // 2)
                current_item_batch_size = max(1024, current_item_batch_size // 2)

                print(
                    f"⚠️ High memory usage ({gpu_memory_usage * 100:.1f}%)! Reduced batch sizes: Users={current_user_batch_size}, Items={current_item_batch_size}")

                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()

            # If memory usage has decreased and we previously reduced batch sizes
            elif gpu_memory_usage < memory_threshold * 0.8 and using_shared_memory:
                # We can try increasing batch sizes again (being conservative)
                using_shared_memory = False

                current_user_batch_size = min(user_batch_size, int(current_user_batch_size * 1.5))
                current_item_batch_size = min(item_batch_size, int(current_item_batch_size * 1.5))

                print(
                    f"🔼 Memory usage has decreased. Increased batch sizes: Users={current_user_batch_size}, Items={current_item_batch_size}")

            return current_user_batch_size, current_item_batch_size

        with torch.no_grad():
            # Process users in dynamically sized batches
            i = 0
            while i < num_users:
                # Adjust batch sizes based on current memory usage
                current_user_batch_size, _ = adjust_batch_sizes()

                # Process the next batch of users
                end_idx = min(i + current_user_batch_size, num_users)
                batch_users_np = users_np[i:end_idx]
                batch_timestamps_np = timestamps_np[i:end_idx]

                batch_size = len(batch_users_np)
                batch_num = i // user_batch_size + 1 if user_batch_size > 0 else 1
                total_batches = (num_users - 1) // user_batch_size + 1 if user_batch_size > 0 else 1

                start_time = time()
                print(f'Processing user batch {batch_num}/{total_batches} ({batch_size} users)')

                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.device)
                timestamps_tensor = torch.tensor(batch_timestamps_np, dtype=torch.float32, device=self.device)

                # Allocate scores tensor
                all_scores = torch.zeros((batch_size, num_items), device=self.device)

                # Process items in dynamically sized batches
                j = 0
                while j < num_items:
                    try:
                        # Adjust item batch size based on current memory usage
                        _, current_item_batch_size = adjust_batch_sizes()

                        # Get current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        batch_items = items[j:end_item_idx]
                        item_batch_size_actual = len(batch_items)

                        # Create input tensors efficiently
                        users_matrix = users_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(-1)
                        items_matrix = batch_items.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                        timestamps_matrix = timestamps_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(
                            -1)

                        # Process metadata features if available
                        if metadata_cache:
                            metadata_matrices = []
                            batch_items_np = batch_items.cpu().numpy()

                            for feature in metadata_features:
                                if feature not in metadata_cache:
                                    continue

                                # Get feature cache
                                feature_cache = metadata_cache[feature]

                                # Get all feature values for this batch of items
                                feature_values = [feature_cache.get(item_id, 0.0) for item_id in batch_items_np]

                                # Convert to tensor efficiently
                                if isinstance(feature_values[0], list):
                                    # For categorical features (lists)
                                    feature_tensor = torch.tensor(feature_values, dtype=torch.float32,
                                                                  device=self.device)

                                    # Repeat each item's features for all users
                                    expanded_tensor = feature_tensor.repeat_interleave(batch_size, dim=0)
                                    metadata_matrices.append(expanded_tensor)
                                else:
                                    # For numeric features
                                    feature_tensor = torch.tensor(feature_values, dtype=torch.float32,
                                                                  device=self.device)

                                    # Expand to match dimensions
                                    expanded_tensor = feature_tensor.unsqueeze(0).expand(batch_size, -1).reshape(-1, 1)
                                    metadata_matrices.append(expanded_tensor)
                        else:
                            metadata_matrices = []

                        # Make predictions
                        batch_predictions = self.model(users_matrix, items_matrix, timestamps_matrix, metadata_matrices)

                        # Reshape predictions to user x item matrix
                        batch_predictions = batch_predictions.view(batch_size, item_batch_size_actual)

                        # Store in the appropriate slice of the scores tensor
                        all_scores[:, j:end_item_idx] = batch_predictions

                        # Clean up to save memory
                        del users_matrix, items_matrix, timestamps_matrix, batch_predictions, metadata_matrices

                        # Move to next item batch
                        j = end_item_idx

                    except RuntimeError as e:
                        # If we hit CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            # Drastically reduce item batch size and retry
                            current_item_batch_size = max(128, current_item_batch_size // 4)
                            using_shared_memory = True  # Mark as using shared memory

                            print(f"🚨 CUDA OOM error! Reduced item batch size to {current_item_batch_size}")

                            # Clear memory and retry
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            # If it's another error, re-raise it
                            raise e

                # Process and store predictions for each user in this batch
                for idx, user_id in enumerate(batch_users_np):
                    # Get scores for this user
                    user_scores = all_scores[idx].cpu().numpy()

                    # Find top-k items - faster with partial sort
                    top_k_indices = np.argpartition(user_scores, -k)[-k:]
                    top_k_indices = top_k_indices[np.argsort(-user_scores[top_k_indices])]

                    # Store only top-k indices
                    predictions[user_id] = top_k_indices.tolist()

                # Clear batch data
                del all_scores, users_tensor, timestamps_tensor
                torch.cuda.empty_cache()
                gc.collect()

                # Report timing
                elapsed = time() - start_time
                remaining_users = num_users - end_idx
                remaining_batches = remaining_users / batch_size if batch_size > 0 else 0
                eta = remaining_batches * elapsed

                print(f"Batch processed in {elapsed:.2f} seconds. ETA: {eta:.2f} seconds")

                # Move to next user batch
                i = end_idx

        return predictions

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_users': len(self.unique_users),
                'num_items': len(self.unique_items),
                'factors': self.model.mf_user_embedding.weight.shape[1],
                'layers': [layer.out_features for layer in self.model.mlp_layers],
                'time_factors': self.time_factors,
                'dropout': self.model.dropout
            }
        }, path)
    
    @classmethod
    def load(cls, path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        # Create model
        recommender = cls(
            unique_users=list(range(config['num_users'])),
            unique_items=list(range(config['num_items'])),
            factors=config['factors'],
            layers=config['layers'],
            time_factors=config['time_factors'],
            dropout=config.get('dropout', 0.0),
            device=device
        )
        
        # Load weights
        recommender.model.load_state_dict(checkpoint['model_state_dict'])
        
        return recommender