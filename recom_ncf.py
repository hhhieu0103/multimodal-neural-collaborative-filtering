import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF
import numpy as np
import pandas as pd
import gc
import time
import math
from helpers.image_dataloader import ImageDataLoader

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
        mlp_feature_dims=None, # Dictionary where keys are features, values are tuples (input, output)
        image_dim=None,
        image_dataloader: ImageDataLoader=None,
        df_features: pd.DataFrame=None,
        num_mlp_layers=4,
        layers_ratio=2,
        learning_rate=0.001,
        epochs=20,
        dropout=0.0,
        weight_decay=0.0,
        loss_fn='bce',
        optimizer='sgd',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # device=torch.device("cpu"),
    ):
        self.feature_values = None
        self.val_losses = None
        self.train_losses = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.early_stopping_patience = 3
        self.mlp_feature_dims = mlp_feature_dims
        self.image_dim = image_dim
        self.image_dataloader = image_dataloader

        if self.mlp_feature_dims is not None:
            if df_features is None:
                raise ValueError('The model is using additional features, but df_features is None')
            else:
                df_features = df_features.copy()
                df_features.set_index('item_id', inplace=True)
                self.feature_values = df_features.to_dict(orient='series')

        self.unique_users = torch.tensor(unique_users)
        self.unique_items = torch.tensor(unique_items)

        self.model = NCF(
            num_users=len(unique_users),
            num_items=len(unique_items),
            factors=factors,
            mlp_user_item_dim=mlp_user_item_dim,
            mlp_feature_dims=mlp_feature_dims,
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
            users, items, ratings, features, images = self._move_data_to_device(data_batch)

            optimizer.zero_grad()

            predictions = self.model(users, items, features, images)
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
                users, items, ratings, features, images = self._move_data_to_device(data_batch)
                
                predictions = self.model(users, items, features, images)
                loss = loss_fn(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches

    def _move_data_to_device(self, data_batch):
        users, items, ratings, features, images = data_batch

        users = users.to(self.device)
        items = items.to(self.device)
        ratings = ratings.to(self.device)

        if self.mlp_feature_dims is not None and len(features) > 0:
            for feature, (input_dim, output_dim) in self.mlp_feature_dims.items():
                if input_dim == 1:
                    features[feature] = features[feature].to(self.device)
                else:
                    indices, offsets = features[feature]
                    indices = indices.to(self.device)
                    offsets = offsets.to(self.device)
                    features[feature] = (indices, offsets)
        else:
            features = None

        if self.image_dim is not None and len(images) > 0:
            images = images.to(self.device)
        else:
            images = None

        return users, items, ratings, features, images

    def predict(
            self,
            users: np.ndarray,
            items: np.ndarray,
            feature_tensors: dict | None = None,
            image_tensors = None
    ):
        self.model.eval()

        num_users = len(users)
        num_items = len(items)

        users_tensor = torch.tensor(users, dtype=torch.long, device=self.device)
        items_tensor = torch.tensor(items, dtype=torch.long, device=self.device)

        users_tensor = users_tensor.unsqueeze(1).expand(-1, num_items).reshape(-1)
        items_tensor = items_tensor.unsqueeze(0).expand(num_users, -1).reshape(-1)

        if self.feature_values is not None and feature_tensors is None:
            feature_tensors = self._get_batch_feature_tensors(items, num_users)

        if image_tensors is not None:
            image_tensors = image_tensors.repeat(num_users, 1)

        with torch.no_grad():
            predictions = self.model(users_tensor, items_tensor, feature_tensors, image_tensors)

        return predictions.view(num_users, num_items)

    def batch_predict_for_users(
            self,
            users: np.ndarray,
            items: np.ndarray | None = None,
            k=50,
            user_batch_size=128,
            item_batch_size=1024,
    ):
        """
        Generate predictions for all items for each user, with dynamic batch size adjustment
        to optimize GPU memory usage.

        Args:
            users: List or array of user IDs
            items: Optional tensor of item IDs. If None, all items are used
            k: Number of top items to retrieve per user
            user_batch_size: Initial number of users to process in each batch
            item_batch_size: Initial number of items to process in each batch

        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        predictions = {}

        # Process input data to standard formats
        if items is None:
            items = np.array(list(range(len(self.unique_items))))

        # Get dimensions
        num_users = len(users)
        num_items = len(items)

        # Initialize batch sizes (will be adjusted dynamically)
        current_user_batch_size = user_batch_size
        current_item_batch_size = item_batch_size

        prev_item_batch_size = None
        feature_tensors_cache = None
        item_batch_idx = 0

        # print(f"Processing predictions for {num_users} users and {num_items} items")

        with torch.no_grad():
            i = 0
            while i < num_users:
                # Get the next batch of users
                end_idx = min(i + current_user_batch_size, num_users)
                user_batch = users[i:end_idx]
                actual_user_batch_size = len(user_batch)

                # Log progress
                print(f'Processing {i + 1} of {num_users} users... ({i / num_users:.2%})')

                # Initialize tensor to store all scores for this batch
                all_scores = torch.zeros((actual_user_batch_size, num_items), device=self.device)

                if self.mlp_feature_dims is not None and feature_tensors_cache is None and current_item_batch_size == prev_item_batch_size:
                    feature_tensors_cache = self._build_feature_tensors_cache(items, current_item_batch_size,
                                                                              actual_user_batch_size)

                # Process items in batches for the current user batch
                j = 0
                while j < num_items:
                    try:
                        # Get the current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        item_batch = items[j:end_item_idx]

                        feature_tensors = None
                        if feature_tensors_cache is not None and actual_user_batch_size == current_user_batch_size:
                            feature_tensors = feature_tensors_cache[item_batch_idx]
                            item_batch_idx += 1

                        image_tensor = None
                        if self.image_dim is not None:
                            image_tensor = self.image_dataloader.get_batch_tensors(item_batch)

                        batch_scores = self.predict(user_batch, item_batch, feature_tensors, image_tensor)

                        # Store scores in the all_scores tensor
                        all_scores[:, j:end_item_idx] = batch_scores

                        # Move to next item batch
                        j = end_item_idx

                    except RuntimeError as e:
                        # Handle OOM errors by reducing batch size and retrying
                        if "CUDA out of memory" in str(e):
                            current_user_batch_size, current_item_batch_size = self._handle_oom_error(
                                current_user_batch_size, current_item_batch_size)
                            # Don't advance the index - retry with smaller batch
                            continue
                        else:
                            # Re-raise other errors
                            raise e

                # Extract top-k items for each user in the batch
                self._extract_top_k_items(user_batch, all_scores, k, predictions)

                item_batch_idx = 0
                if prev_item_batch_size != current_item_batch_size:
                    prev_item_batch_size = current_item_batch_size
                    current_user_batch_size, current_item_batch_size = self._adjust_batch_size(current_user_batch_size,
                                                                                               current_item_batch_size)

                # Clean up batch memory
                del all_scores
                torch.cuda.empty_cache()
                gc.collect()

                # Move to next user batch
                i = end_idx

        return predictions

    # Helper methods for batch_predict_for_users

    def _build_feature_tensors_cache(self, items, item_batch_size, num_users):
        num_batches = math.ceil(len(items) / item_batch_size)
        feature_tensors_cache = {}
        for batch_idx in range(num_batches):
            batch_items = items[batch_idx * item_batch_size: (batch_idx + 1) * item_batch_size]
            feature_tensors_cache[batch_idx] = self._get_batch_feature_tensors(batch_items, num_users)
        return feature_tensors_cache

    def _get_batch_feature_tensors(self, items, num_users):
        feature_tensors = {}

        # Process each feature type
        for feature, (input_dim, output_dim) in self.mlp_feature_dims.items():
            if input_dim == 1:
                # For numeric features - more efficient handling
                feature_values = np.array(self.feature_values[feature][items])
                tensor = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
                feature_tensors[feature] = tensor.unsqueeze(0).expand(num_users, -1).reshape(-1)
            else:
                all_indices = []
                all_offsets = [0]

                for item in items:
                    indices = self.feature_values[feature][item]
                    all_indices.extend(indices)
                    all_offsets.append(all_offsets[-1] + len(indices))

                # Remove the last offset (we only need num_items offsets)
                all_offsets.pop()

                # Convert to tensors
                indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=self.device)

                # Now repeat for each user
                repeated_indices = indices_tensor.repeat(num_users)

                # For offsets, we need to add the proper offset for each user
                user_offsets = []
                last_offset = len(all_indices)  # Total number of indices for one user's items

                for u in range(num_users):
                    user_offsets.extend([offset + u * last_offset for offset in all_offsets])

                offsets_tensor = torch.tensor(user_offsets, dtype=torch.long, device=self.device)

                feature_tensors[feature] = (repeated_indices, offsets_tensor)
        return feature_tensors

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
        is_stable = False

        if gpu_memory_usage >= 0.9:

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

        elif gpu_memory_usage < 0.75:

            # increasing_rate = max(1.0, 0.95 / gpu_memory_usage / self.total_emb_dims * 256)
            increasing_rate = 1.1

            new_user_batch_size = round(current_user_batch_size * increasing_rate)
            new_item_batch_size = round(current_item_batch_size * increasing_rate)

            print('Memory usage:', gpu_memory_usage ,'. Increasing batch size with increasing rate of', increasing_rate)

            if new_user_batch_size != current_user_batch_size:
                print('Increased user batch size from', current_user_batch_size, 'to', new_user_batch_size)
                current_user_batch_size = new_user_batch_size

            if new_item_batch_size != current_item_batch_size:
                print('Increased item batch size from', current_item_batch_size, 'to', new_item_batch_size)
                current_item_batch_size = new_item_batch_size

        return current_user_batch_size, current_item_batch_size

    def _handle_oom_error(self, current_user_batch_size, current_item_batch_size):
        """Handle out of memory error by reducing batch size"""
        reduced_user_batch_size = max(128, current_user_batch_size // 4)
        reduced_item_batch_size = max(128, current_item_batch_size // 4)

        print(f"CUDA OOM error!")
        print(f"Reduced user batch size to {reduced_user_batch_size}")
        print(f"Reduced item batch size to {reduced_item_batch_size}")

        # Clear memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return reduced_user_batch_size, reduced_item_batch_size

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
