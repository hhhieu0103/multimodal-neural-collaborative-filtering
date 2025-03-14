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
        batch_size=4096,
        epochs=20,
        dropout=0.0,
        weight_decay=0.0,
        loss_fn='bce',
        optimizer='sgd',
        early_stopping_patience=5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
    
    def predict_for_users(self, users, batch_size=1024, items: torch.Tensor | np.ndarray | None = None):
        """
        Generate predictions for all items for each user
        
        Args:
            users: List of user IDs
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary mapping user IDs to tensors of scores for all items
        """
        self.model.eval()
        predictions = {}
        
        if items is None:
            items = torch.arange(len(self.unique_items))
        elif not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long)
        
        with torch.no_grad():
            # Process each user
            for user_id in users:
                user_tensor = torch.full((len(items),), user_id, dtype=torch.long)
                
                all_scores = []
                for i in range(0, len(items), batch_size):
                    batch_items = items[i:i+batch_size]
                    batch_users = user_tensor[i:i+batch_size]
                    
                    # Move to device
                    batch_users = batch_users.to(self.device)
                    batch_items = batch_items.to(self.device)
                    
                    # Get predictions
                    batch_scores = self.model(batch_users, batch_items)
                    all_scores.append(batch_scores.cpu())
                
                # Combine all batches
                user_predictions = torch.cat(all_scores).squeeze()
                predictions[user_id] = user_predictions
                
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