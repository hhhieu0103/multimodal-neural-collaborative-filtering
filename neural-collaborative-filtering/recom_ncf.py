import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF
import numpy as np

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
        layers=[64, 32, 16, 8],
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
        
        self.model = NCF(len(unique_users), len(unique_items), factors, layers, dropout).to(self.device)
        
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
        
        for users, items, ratings in train_data:
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)

            optimizer.zero_grad()
            predictions = self.model(users, items)
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
            for users, items, ratings in val_data:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                
                predictions = self.model(users, items)
                loss = loss_fn(predictions, ratings)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def predict(self, users: torch.Tensor | np.ndarray, items: torch.Tensor | np.ndarray):
        self.model.eval()
        
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long)
        if not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long)
            
        users = users.to(self.device)
        items = items.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(users, items)
            
        return predictions.cpu()
    
    def batch_predict_for_users(self, users: torch.Tensor | np.ndarray, items: torch.Tensor | np.ndarray | None =None, 
                         user_batch_size=1024, item_batch_size=8192, k=10, memory_threshold=0.9):
        """
        Generate predictions for all items for each user with dynamic batch size adjustment
        based on GPU memory usage to avoid expensive shared memory operations
        
        Args:
            users: List or array of user IDs
            items: Optional tensor of item IDs. If None, all items are used.
            user_batch_size: Initial number of users to process in each batch
            item_batch_size: Initial number of items to process in each batch
            k: Number of top items to retain per user
            memory_threshold: GPU memory utilization threshold (0-1) to trigger batch size reduction
            
        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        import gc
        
        self.model.eval()
        predictions = {}
        
        # Convert users to numpy for iteration if needed
        if isinstance(users, torch.Tensor):
            users_np = users.cpu().numpy()
        else:
            users_np = np.array(users)
        
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
        
        # Function to check GPU memory usage
        def get_gpu_memory_usage():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                # Use reserved memory to calculate usage since it better reflects
                # the total memory footprint including allocated memory and caching
                gpu_memory_usage = reserved / 8.0  # 8GB is the dedicated GPU memory  
                return reserved, gpu_memory_usage
            return 0, 0, 0
        
        # Function to adjust batch sizes based on memory usage
        def adjust_batch_sizes():
            nonlocal current_user_batch_size, current_item_batch_size, using_shared_memory
            
            reserved, gpu_memory_usage = get_gpu_memory_usage()
            
            # Print memory stats
            print(f"GPU Memory: Reserved={reserved:.2f}GB, Usage={gpu_memory_usage:.2f}")
            
            # If memory usage is high but we haven't reduced batch sizes yet
            if gpu_memory_usage > memory_threshold and not using_shared_memory:
                # We're likely using shared memory
                using_shared_memory = True
                
                # Reduce batch sizes by 50%
                current_user_batch_size = max(32, current_user_batch_size // 2)
                current_item_batch_size = max(1024, current_item_batch_size // 2)
                
                print(f"High memory usage detected! Reduced batch sizes: users={current_user_batch_size}, items={current_item_batch_size}")
                
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
            
            # If memory usage has decreased and we previously reduced batch sizes
            elif gpu_memory_usage < memory_threshold * 0.7 and using_shared_memory:
                # We can try increasing batch sizes again (being conservative)
                using_shared_memory = False
                
                # Increase batch sizes, but not above initial values
                current_user_batch_size = min(user_batch_size, int(current_user_batch_size * 1.5))
                current_item_batch_size = min(item_batch_size, int(current_item_batch_size * 1.5))
                
                print(f"Memory usage has decreased. Increased batch sizes: users={current_user_batch_size}, items={current_item_batch_size}")
            
            return int(current_user_batch_size), int(current_item_batch_size)
        
        with torch.no_grad():
            # Process users in dynamically sized batches
            i = 0
            while i < num_users:
                # Process the next batch of users with current batch size
                end_idx = min(i + current_user_batch_size, num_users)
                batch_users_np = users_np[i:end_idx]
                batch_size = len(batch_users_np)
                batch_num = i // current_user_batch_size + 1
                total_batches = (num_users - 1) // current_user_batch_size + 1
                
                print(f'Processing user batch {batch_num}/{total_batches} ({batch_size} users)')
                
                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.device)
                
                # Allocate scores tensor
                all_scores = torch.zeros((batch_size, num_items), device=self.device)
                
                # Process items in dynamically sized batches
                j = 0
                current_item_batch_size = item_batch_size  # Initialize to the starting value
                while j < num_items:
                    try:
                        # Get current item batch
                        end_item_idx = min(j + current_item_batch_size, num_items)
                        batch_items = items[j:end_item_idx]
                        item_batch_size_actual = len(batch_items)
                        
                        item_batch_num = j // current_item_batch_size + 1
                        total_item_batches = (num_items - 1) // current_item_batch_size + 1
                        
                        if batch_num == 1:  # Only print for first user batch
                            print(f'  - Item batch {item_batch_num}/{total_item_batches}')
                        
                        # Create user-item pairs
                        users_matrix = users_tensor.repeat_interleave(item_batch_size_actual)
                        items_matrix = batch_items.repeat(batch_size)
                        
                        # Make predictions
                        batch_scores = self.model(users_matrix, items_matrix)
                        
                        # Reshape and store scores
                        batch_scores = batch_scores.view(batch_size, item_batch_size_actual)
                        all_scores[:, j:end_item_idx] = batch_scores
                        
                        # Measure memory usage AFTER predictions are made and BEFORE cleanup
                        # This gives us the most accurate picture of peak memory usage
                        _, item_batch = adjust_batch_sizes()
                        current_item_batch_size = item_batch
                        
                        # Clean up intermediate tensors
                        del users_matrix, items_matrix, batch_scores
                        
                        # Move to next item batch
                        j = end_item_idx
                        
                    except RuntimeError as e:
                        # If we hit CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            # Drastically reduce item batch size and retry
                            current_item_batch_size = max(128, current_item_batch_size // 4)
                            print(f"CUDA OOM error! Reduced item batch size to {current_item_batch_size}")
                            
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
                del all_scores, users_tensor
                torch.cuda.empty_cache()
                gc.collect()
                
                # Measure memory and adjust user batch size AFTER processing a complete user batch
                user_batch, _ = adjust_batch_sizes()
                current_user_batch_size = user_batch
                
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
            dropout=config.get('dropout', 0.0),
            device=device
        )
        
        # Load weights
        recommender.model.load_state_dict(checkpoint['model_state_dict'])
        
        return recommender