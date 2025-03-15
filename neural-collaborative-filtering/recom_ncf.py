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
    
    def fit(self, train_data: DataLoader, eval_data: DataLoader):
        train_losses = []
        eval_losses = []
        
        best_eval_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        loss_fn = self.get_loss_function()
        optimizer = self.get_optimizer()
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train(train_data, loss_fn, optimizer)
            train_losses.append(train_loss)
            
            eval_loss = self.evaluate(eval_data, loss_fn)
            eval_losses.append(eval_loss)
            
            print(f'Train loss: {train_loss:.6f}, Eval loss: {eval_loss:.6f}')
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
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
        self.eval_losses = eval_losses
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
    
    def evaluate(self, eval_data, loss_fn):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for users, items, ratings in eval_data:
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
    
    def batch_predict_for_users(self, users, items=None, user_batch_size=1024, item_batch_size=8192, k=10):
        """
        Generate predictions for all items for each user and return the top-k items
        
        Args:
            users: List or array of user IDs
            items: Optional tensor of item IDs. If None, all items are used.
            user_batch_size: Number of users to process in each batch
            item_batch_size: Number of items to process in each batch
            k: Number of top items to retain per user
            
        Returns:
            Dictionary mapping user IDs to lists of top-k item indices
        """
        import gc  # For garbage collection
        
        self.model.eval()
        predictions = {}
        
        # Convert users to numpy for final processing
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
        num_batches = (num_users - 1) // user_batch_size + 1
        
        with torch.no_grad():
            # Process users in batches
            for i in range(0, num_users, user_batch_size):
                batch_users_np = users_np[i:i+user_batch_size]
                batch_size = len(batch_users_np)
                batch_num = i // user_batch_size + 1
                
                print(f'Processing user batch {batch_num}/{num_batches} ({batch_size} users)')
                
                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.device)
                
                # Allocate scores tensor for this batch
                all_scores_gpu = torch.zeros((batch_size, num_items), device=self.device)
                
                # Process items in batches
                for j in range(0, num_items, item_batch_size):
                        
                    batch_items = items[j:j+item_batch_size]
                    item_batch_size_actual = len(batch_items)
                    
                    # Create user-item pairs
                    users_matrix = users_tensor.repeat_interleave(item_batch_size_actual)
                    items_matrix = batch_items.repeat(batch_size)
                    
                    # Make predictions
                    batch_scores = self.model(users_matrix, items_matrix)
                    
                    # Reshape and store scores
                    batch_scores = batch_scores.view(batch_size, item_batch_size_actual)
                    all_scores_gpu[:, j:j+item_batch_size_actual] = batch_scores
                    
                    # Clean up intermediate tensors
                    del users_matrix, items_matrix, batch_scores
                
                # Store predictions for this batch of users
                for idx, user_id in enumerate(batch_users_np):
                    # Get the user's scores and convert to numpy right away
                    user_scores = all_scores_gpu[idx].cpu().numpy()
                    
                    # Find top-k items using argpartition (much faster than argsort for just finding top-k)
                    top_k_indices = np.argpartition(user_scores, -k)[-k:]
                    # Sort just the top k items by score
                    top_k_indices = top_k_indices[np.argsort(-user_scores[top_k_indices])]
                    
                    # Store only the top-k indices, not the full score array
                    predictions[user_id] = top_k_indices.tolist()
                    
                    # Clean up user scores to free memory immediately
                    del user_scores
                
                # Clean up
                del all_scores_gpu, users_tensor
                torch.cuda.empty_cache()
                gc.collect()
        
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