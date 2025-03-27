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
        min_memory_target=0.85,  # Target minimum memory usage (85%)
        max_memory_target=0.95,  # Maximum memory threshold (95%)
        initial_item_batch=1024  # Starting item batch size
    ):
        """
        Generate predictions for all items for each user with fixed user batch size
        and initially adjusted item batch size based on GPU memory usage. Once a
        stable batch size is found, it will be used for all subsequent processing.
        
        Args:
            users: List or array of user IDs
            timestamps: Timestamps for each user
            df_metadata: DataFrame containing metadata features for items
            metadata_features: List of metadata feature column names to use
            items: Optional tensor of item IDs. If None, all items are used.
            k: Number of top items to retain per user
            min_memory_target: Target minimum memory utilization (0-1)
            max_memory_target: Maximum memory utilization threshold (0-1)
            initial_item_batch: Starting item batch size
            
        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        import gc
        
        self.model.eval()
        predictions = {}
        
        # Fixed user batch size
        USER_BATCH_SIZE = 1024
        
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
        
        # Item batch size will be adjusted dynamically, starting with the initial value
        current_item_batch_size = initial_item_batch
        
        # Flag to indicate if we've found a stable batch size
        stable_batch_size_found = False
        
        # Estimate total GPU memory (default to 8GB if CUDA not available)
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        else:
            total_gpu_memory = 8.0  # Default to 8GB
        
        print(f"Total GPU memory: {total_gpu_memory:.2f} GB")
        print(f"Target memory usage: {min_memory_target*100:.0f}%-{max_memory_target*100:.0f}% ({min_memory_target*total_gpu_memory:.2f}-{max_memory_target*total_gpu_memory:.2f} GB)")
        
        # Track consecutive increases/decreases for smarter adjustments
        consecutive_increases = 0
        consecutive_decreases = 0
        
        # Create a cache for metadata lookups to improve performance
        metadata_cache = {}
        
        def get_gpu_memory_usage():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                gpu_memory_usage = reserved / total_gpu_memory
                return reserved, allocated, gpu_memory_usage
            return 0, 0, 0
        
        # Our item batch size adjustment function that will only be called until we find a stable size
        def find_optimal_batch_size():
            nonlocal current_item_batch_size, consecutive_increases, consecutive_decreases, stable_batch_size_found
            
            reserved, allocated, gpu_memory_usage = get_gpu_memory_usage()
            
            # Print detailed memory info
            print(f"GPU Memory: Reserved={reserved:.2f}GB ({gpu_memory_usage*100:.1f}%), Allocated={allocated:.2f}GB, Item batch size: {current_item_batch_size}")
            
            # Store batch size before adjustment for reporting
            old_batch_size = current_item_batch_size
            
            # Check if we've found a stable batch size (memory usage in target range)
            if min_memory_target <= gpu_memory_usage <= max_memory_target:
                stable_batch_size_found = True
                print(f"✓ Found stable batch size: {current_item_batch_size} with memory usage {gpu_memory_usage*100:.1f}%")
                print(f"ℹ️ Will use this batch size for all remaining processing")
                return current_item_batch_size
            
            # Memory is too high - reduce batch size
            if gpu_memory_usage > max_memory_target:
                # Calculate reduction factor based on how much we're over
                reduction_factor = 0.8  # Default 20% reduction
                
                # If we're significantly over the threshold, reduce more aggressively
                if gpu_memory_usage > 0.98:  # Almost out of memory
                    reduction_factor = 0.5  # 50% reduction
                elif gpu_memory_usage > 0.97:
                    reduction_factor = 0.6  # 40% reduction
                elif gpu_memory_usage > 0.96:
                    reduction_factor = 0.7  # 30% reduction
                
                current_item_batch_size = max(256, int(current_item_batch_size * reduction_factor))
                consecutive_decreases += 1
                consecutive_increases = 0
                
                print(f"⚠️ High memory usage ({gpu_memory_usage*100:.1f}%)! Reduced item batch size: {old_batch_size} → {current_item_batch_size}")
                
                # Force clear cache if we're getting close to OOM
                if gpu_memory_usage > 0.97:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Memory is too low - increase batch size
            elif gpu_memory_usage < min_memory_target:
                # Calculate increase factor based on how far below target we are
                increase_factor = 1.2  # Default 20% increase
                
                # If we're significantly below the target, increase more aggressively
                if gpu_memory_usage < 0.6:
                    increase_factor = 2.0  # Double the batch size
                elif gpu_memory_usage < 0.7:
                    increase_factor = 1.5  # 50% increase
                elif gpu_memory_usage < 0.8:
                    increase_factor = 1.3  # 30% increase
                
                # Be more conservative with increases if we've seen high memory recently
                if consecutive_decreases > 0:
                    increase_factor = min(increase_factor, 1.2)
                
                # If we've been steadily increasing without issues, be more aggressive
                if consecutive_increases > 2:
                    increase_factor = min(2.0, increase_factor * 1.2)
                
                # Apply the increase
                current_item_batch_size = min(16384, int(current_item_batch_size * increase_factor))
                consecutive_increases += 1
                consecutive_decreases = 0
                
                print(f"🔼 Low memory usage ({gpu_memory_usage*100:.1f}%)! Increased item batch size: {old_batch_size} → {current_item_batch_size}")
            
            return current_item_batch_size
        
        with torch.no_grad():
            # Process users in fixed batch size
            i = 0
            while i < num_users:
                # Process the next batch of users
                end_idx = min(i + USER_BATCH_SIZE, num_users)
                batch_users_np = users_np[i:end_idx]
                batch_timestamps_np = timestamps_np[i:end_idx]
                
                batch_size = len(batch_users_np)
                batch_num = i // USER_BATCH_SIZE + 1
                total_batches = (num_users - 1) // USER_BATCH_SIZE + 1
                
                print(f'Processing user batch {batch_num}/{total_batches} ({batch_size} users)')
                
                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.device)
                timestamps_tensor = torch.tensor(batch_timestamps_np, dtype=torch.float32, device=self.device)
                
                # Allocate scores tensor
                all_scores = torch.zeros((batch_size, num_items), device=self.device)
                
                # Process items in dynamically sized batches
                j = 0
                while j < num_items:
                    try:
                        # Get current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        batch_items = items[j:end_item_idx]
                        item_batch_size_actual = len(batch_items)
                        
                        users_matrix = users_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(-1)
                        items_matrix = batch_items.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                        timestamps_matrix = timestamps_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(-1)
                        
                        # Metadata handling with optimized caching
                        if df_metadata is not None and not df_metadata.empty and metadata_features:
                            metadata_matrices = []
                            batch_items_np = batch_items.cpu().numpy()
                            
                            for feature in metadata_features:
                                # Use cached mappings if available
                                if feature not in metadata_cache:
                                    # Create and cache the item_id to feature mapping
                                    metadata_cache[feature] = {}
                                    for item_id, feature_value in zip(df_metadata.index, df_metadata[feature]):
                                        metadata_cache[feature][item_id] = feature_value
                                
                                # Get the cached mapping
                                feature_map = metadata_cache[feature]
                                
                                # Get feature values for items in this batch
                                feature_values = [feature_map.get(item_id, 0.0) for item_id in batch_items_np]
                                
                                # Create tensor and expand to match user dimensions
                                feature_tensor = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
                                feature_matrix = feature_tensor.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                                metadata_matrices.append(feature_matrix)
                        else:
                            metadata_matrices = []
                        
                        # Make predictions
                        batch_scores = self.model(users_matrix, items_matrix, timestamps_matrix, metadata_matrices)
                        
                        # Reshape and store scores
                        batch_scores = batch_scores.view(batch_size, item_batch_size_actual)
                        all_scores[:, j:end_item_idx] = batch_scores
                        
                        # Clean up intermediate tensors
                        del users_matrix, items_matrix, timestamps_matrix, metadata_matrices, batch_scores
                        
                        # Only adjust batch size if there are more items to process AND we haven't found a stable size
                        if j + item_batch_size_actual < num_items and not stable_batch_size_found:
                            current_item_batch_size = find_optimal_batch_size()
                        elif stable_batch_size_found and j == 0:  # First item batch of a user batch, after stable size is found
                            # Just report memory usage without adjusting
                            reserved, allocated, gpu_memory_usage = get_gpu_memory_usage()
                            print(f"✓ Using stable batch size: {current_item_batch_size}, Memory: {gpu_memory_usage*100:.1f}%")
                        
                        # Move to next item batch
                        j = end_item_idx
                        
                    except RuntimeError as e:
                        # If we hit CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            # Drastically reduce item batch size and retry
                            current_item_batch_size = max(128, current_item_batch_size // 4)
                            consecutive_decreases += 2  # Penalize more for OOM
                            consecutive_increases = 0
                            stable_batch_size_found = False  # Reset stable flag on OOM error
                            print(f"🚨 CUDA OOM error! Reduced item batch size to {current_item_batch_size}")
                            
                            # Clear memory and retry
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            # If it's another error, re-raise it
                            raise e
                
                # Process and store predictions for each user in this batch
                for idx, user_id in enumerate(batch_users_np):
                    # Get scores and convert to numpy
                    user_scores = all_scores[idx].cpu().numpy()
                    
                    # Find top-k items
                    top_k_indices = np.argpartition(user_scores, -k)[-k:]
                    top_k_indices = top_k_indices[np.argsort(-user_scores[top_k_indices])]
                    
                    # Store only top-k indices
                    predictions[user_id] = top_k_indices.tolist()
                    
                    # Clean up
                    del user_scores
                    
                # Clean up batch data
                del all_scores, users_tensor, timestamps_tensor
                torch.cuda.empty_cache()
                gc.collect()
                
                # Move to next user batch
                i = end_idx
        
        return predictions
        
        with torch.no_grad():
            # Process users in fixed batch size
            i = 0
            while i < num_users:
                # Process the next batch of users
                end_idx = min(i + USER_BATCH_SIZE, num_users)
                batch_users_np = users_np[i:end_idx]
                batch_timestamps_np = timestamps_np[i:end_idx]
                
                batch_size = len(batch_users_np)
                batch_num = i // USER_BATCH_SIZE + 1
                total_batches = (num_users - 1) // USER_BATCH_SIZE + 1
                
                print(f'Processing user batch {batch_num}/{total_batches} ({batch_size} users)')
                
                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.device)
                timestamps_tensor = torch.tensor(batch_timestamps_np, dtype=torch.float32, device=self.device)
                
                # Allocate scores tensor
                all_scores = torch.zeros((batch_size, num_items), device=self.device)
                
                # Track if we're on the first item batch for this user batch
                first_item_batch = True
                
                # Process items in dynamically sized batches
                j = 0
                while j < num_items:
                    try:
                        # Get current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        batch_items = items[j:end_item_idx]
                        item_batch_size_actual = len(batch_items)
                        
                        users_matrix = users_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(-1)
                        items_matrix = batch_items.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                        timestamps_matrix = timestamps_tensor.unsqueeze(1).expand(-1, item_batch_size_actual).reshape(-1)
                        
                        # Metadata handling with optimized caching
                        if df_metadata is not None and not df_metadata.empty and metadata_features:
                            metadata_matrices = []
                            batch_items_np = batch_items.cpu().numpy()
                            
                            for feature in metadata_features:
                                # Use cached mappings if available
                                if feature not in metadata_cache:
                                    # Create and cache the item_id to feature mapping
                                    metadata_cache[feature] = {}
                                    for item_id, feature_value in zip(df_metadata.index, df_metadata[feature]):
                                        metadata_cache[feature][item_id] = feature_value
                                
                                # Get the cached mapping
                                feature_map = metadata_cache[feature]
                                
                                # Get feature values for items in this batch
                                feature_values = [feature_map.get(item_id, 0.0) for item_id in batch_items_np]
                                
                                # Create tensor and expand to match user dimensions
                                feature_tensor = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
                                feature_matrix = feature_tensor.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                                metadata_matrices.append(feature_matrix)
                        else:
                            metadata_matrices = []
                        
                        # Make predictions
                        batch_scores = self.model(users_matrix, items_matrix, timestamps_matrix, metadata_matrices)
                        
                        # Reshape and store scores
                        batch_scores = batch_scores.view(batch_size, item_batch_size_actual)
                        all_scores[:, j:end_item_idx] = batch_scores
                        
                        # Clean up intermediate tensors
                        del users_matrix, items_matrix, timestamps_matrix, metadata_matrices, batch_scores
                        
                        # Only adjust batch size if there are more items to process
                        if j + item_batch_size_actual < num_items:
                            # Skip adjustment for the first item batch if we've already found a stable batch size
                            if stable_batch_size_found and first_item_batch:
                                print(f"✓ Skipping adjustment for first item batch - using stable batch size: {current_item_batch_size}")
                                # Still report current memory usage without adjusting
                                reserved, allocated, gpu_memory_usage = get_gpu_memory_usage()
                                print(f"GPU Memory: Reserved={reserved:.2f}GB ({gpu_memory_usage*100:.1f}%), Allocated={allocated:.2f}GB")
                            else:
                                current_item_batch_size = adjust_item_batch_size()
                            
                            # No longer the first item batch
                            first_item_batch = False
                        
                        # Move to next item batch
                        j = end_item_idx
                        
                    except RuntimeError as e:
                        # If we hit CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            # Drastically reduce item batch size and retry
                            current_item_batch_size = max(128, current_item_batch_size // 4)
                            consecutive_decreases += 2  # Penalize more for OOM
                            consecutive_increases = 0
                            stable_batch_size_found = False  # Reset stable flag on OOM error
                            print(f"🚨 CUDA OOM error! Reduced item batch size to {current_item_batch_size}")
                            
                            # Clear memory and retry
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            # If it's another error, re-raise it
                            raise e
                
                # Process and store predictions for each user in this batch
                for idx, user_id in enumerate(batch_users_np):
                    # Get scores and convert to numpy
                    user_scores = all_scores[idx].cpu().numpy()
                    
                    # Find top-k items
                    top_k_indices = np.argpartition(user_scores, -k)[-k:]
                    top_k_indices = top_k_indices[np.argsort(-user_scores[top_k_indices])]
                    
                    # Store only top-k indices
                    predictions[user_id] = top_k_indices.tolist()
                    
                    # Clean up
                    del user_scores
                    
                # Clean up batch data
                del all_scores, users_tensor, timestamps_tensor
                torch.cuda.empty_cache()
                gc.collect()
                
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