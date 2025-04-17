import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF, ModelType
import numpy as np
import pandas as pd
import gc
from helpers.mem_map_dataloader import MemMapDataLoader

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, positive_predictions, negative_predictions):
        loss = -torch.mean(torch.log(torch.sigmoid(positive_predictions - negative_predictions)))
        return loss

def _get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

        # Estimate total GPU memory (adjust based on your GPU)
        total_gpu_memory = 8.0  # RTX 4060 Mobile has around 8GB

        gpu_memory_usage = reserved / total_gpu_memory
        return gpu_memory_usage
    return 0

def _adjust_batch_size(current_user_batch_size, current_item_batch_size, decreasing_rate=0.7, increasing_rate=1.1):
    """Adjust item batch size based on current GPU memory usage"""
    gpu_memory_usage = _get_gpu_memory_usage()

    if gpu_memory_usage >= 0.95:

        new_user_batch_size = round(current_user_batch_size * decreasing_rate)
        new_item_batch_size = round(current_item_batch_size * decreasing_rate)

        print('Memory usage:', gpu_memory_usage)
        print('Decreased user batch size from', current_user_batch_size, 'to', new_user_batch_size)
        print('Decreased item batch size from', current_item_batch_size, 'to', new_item_batch_size)

        current_user_batch_size = new_user_batch_size
        current_item_batch_size = new_item_batch_size

        torch.cuda.empty_cache()

    elif gpu_memory_usage < 0.8:

        new_user_batch_size = round(current_user_batch_size * increasing_rate)
        new_item_batch_size = round(current_item_batch_size * increasing_rate)

        print('Memory usage:', gpu_memory_usage)
        print('Increased user batch size from', current_user_batch_size, 'to', new_user_batch_size)
        print('Increased item batch size from', current_item_batch_size, 'to', new_item_batch_size)

        current_user_batch_size = new_user_batch_size
        current_item_batch_size = new_item_batch_size

    return current_user_batch_size, current_item_batch_size

def _extract_top_k_items(user_batch, all_scores, k, predictions):
    """Extract top-k items for each user from scores"""
    user_scores = all_scores.cpu().numpy()

    for i, user_id in enumerate(user_batch):
        # Find top-k items efficiently
        scores = user_scores[i]
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        # Store results
        predictions[user_id] = top_k_indices.tolist()

class NCFRecommender:
    def __init__(
        self,
        unique_users,
        unique_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_feature_dims=None, # Dictionary where keys are features, values are tuples (input, output)
        image_dim=None,
        image_dataloader: MemMapDataLoader=None,
        df_features: pd.DataFrame=None,
        num_mlp_layers=4,
        layers_ratio=2,
        learning_rate=0.001,
        epochs=20,
        dropout=0.0,
        weight_decay=0.0,
        loss_fn='bce',
        optimizer='sgd',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_type=ModelType.EARLY_FUSION
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
        self.early_stopping_patience = 5
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
            image_dim=image_dim,
            num_mlp_layers=num_mlp_layers,
            layers_ratio=layers_ratio,
            dropout=dropout,
            model_type=model_type
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
                    features[feature] = (indices.to(self.device), offsets.to(self.device))
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

        users_tensor = torch.tensor(users, dtype=torch.int, device=self.device)
        items_tensor = torch.tensor(items, dtype=torch.int, device=self.device)

        users_tensor = users_tensor.unsqueeze(1).expand(-1, num_items).reshape(-1)
        items_tensor = items_tensor.unsqueeze(0).expand(num_users, -1).reshape(-1)

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
            batch_increasing_rate=1.1,
            batch_decreasing_rate=0.7,
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
            batch_increasing_rate: The rate at which batch size increases during memory availability
            batch_decreasing_rate: The rate at which batch size decreases during memory shortage
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

        prev_batch_size = (None, None)

        with torch.no_grad():
            i = 0
            while i < num_users:
                end_idx = min(i + current_user_batch_size, num_users)
                user_batch = users[i:end_idx]
                actual_user_batch_size = len(user_batch)
                is_stable = prev_batch_size[0] == current_user_batch_size and prev_batch_size[1] == current_item_batch_size

                print(f'Processing {i + 1} of {num_users} users... ({i / num_users:.2%})')

                all_scores = torch.zeros((actual_user_batch_size, num_items))

                j = 0
                item_batch_idx = 0
                while j < num_items:
                    try:
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        item_batch = items[j:end_item_idx]

                        feature_tensors = None
                        if self.mlp_feature_dims is not None:
                            feature_tensors = self._get_batch_feature_tensors(item_batch, actual_user_batch_size)

                        image_tensor = None
                        if self.image_dim is not None:
                            image_tensor = self.image_dataloader.get_batch_tensors(item_batch)

                        # Process the current batch
                        batch_scores = self.predict(user_batch, item_batch, feature_tensors, image_tensor)

                        all_scores[:, j:end_item_idx] = batch_scores.cpu()

                        item_batch_idx += 1
                        j = end_item_idx

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            current_user_batch_size = max(128, current_user_batch_size // 4)
                            current_item_batch_size = max(128, current_item_batch_size // 4)
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e

                _extract_top_k_items(user_batch, all_scores, k, predictions)

                if not is_stable:
                    prev_batch_size = current_user_batch_size, current_item_batch_size
                    current_user_batch_size, current_item_batch_size = _adjust_batch_size(
                        current_user_batch_size=current_user_batch_size,
                        current_item_batch_size=current_item_batch_size,
                        increasing_rate=batch_increasing_rate,
                        decreasing_rate=batch_decreasing_rate,
                    )

                i = end_idx

        return predictions

    # Helper methods for batch_predict_for_users

    def _get_batch_feature_tensors(self, items, num_users):
        feature_tensors = {}

        # Process each feature type
        for feature, (input_dim, output_dim) in self.mlp_feature_dims.items():
            if input_dim == 1:
                feature_values = np.array(self.feature_values[feature][items])
                feature_tensors[feature] = torch.tensor(feature_values, dtype=torch.float32, device=self.device).repeat(num_users)
            else:
                all_indices = []
                all_offsets = [0]

                for item in items:
                    indices = self.feature_values[feature][item]
                    all_indices.extend(indices)
                    all_offsets.append(all_offsets[-1] + len(indices))

                all_offsets.pop()

                last_offset = len(all_indices)  # Total number of indices for one user's items
                user_indices = np.arange(num_users)[:, np.newaxis]
                offset_array = np.array(all_offsets)[np.newaxis, :]
                user_offsets = (offset_array + user_indices * last_offset).flatten()

                all_indices = torch.tensor(all_indices, dtype=torch.int, device=self.device).repeat(num_users)
                user_offsets = torch.tensor(user_offsets, dtype=torch.long, device=self.device)

                feature_tensors[feature] = (all_indices, user_offsets)
        return feature_tensors
