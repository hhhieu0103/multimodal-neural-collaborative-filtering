import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ncf import NCF
import datetime
import os

class NCFRecommender():
    def __init__(
        self,
        unique_users,
        unique_items,
        factors=8,
        layers=[64, 32, 16, 8],
        learning_rate=0.001,
        batch_size=256,
        epochs=20,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        loss_fn='bce',
        optimizer='sgd'
        ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.unique_users = torch.tensor(unique_users)
        self.unique_items = torch.tensor(unique_items)
        self.model = NCF(len(unique_users), len(unique_items), factors, layers).to(self.device)
        
    def get_loss_function(self):
        if (self.loss_fn == 'bce'):
            return nn.BCELoss()
        raise ValueError(self.loss_fn)
    
    def get_optimizer(self):
        if (self.optimizer == 'sgd'):
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        raise ValueError
    
    def fit(self, train_data: DataLoader, eval_data: DataLoader):
        train_losses = []
        eval_losses = []
        loss_fn = self.get_loss_function()
        optimizer = self.get_optimizer()
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_losses.append(self.train(train_data, loss_fn, optimizer))
            eval_losses.append(self.evaluate(eval_data, loss_fn))
        self.train_losses = train_losses
        self.eval_losses = eval_losses
        print("Done!")
    
    def train(self, train_data, loss_fn, optimizer):
        size = len(train_data.dataset)
        self.model.to(self.device)
        losses = []
        self.model.train()
        for batch, (users, items, ratings) in enumerate(train_data):
            users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)

            optimizer.zero_grad()
            pred = self.model(users, items)
            loss = loss_fn(pred, ratings)

            loss.backward()
            optimizer.step()

            # Add batch's loss to the losses list
            losses.append(loss.item())
            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(ratings)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Return the average loss of an epoch
        return sum(losses)/len(losses)
    
    def evaluate(self, eval_data, loss_fn):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for users, items, ratings in eval_data:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                pred = self.model(users, items)
                loss = loss_fn(pred, ratings)
                losses.append(loss.item())
                
        avg_loss = sum(losses)/len(losses)
        print(f"Avg loss on test: {avg_loss:>8f} \n")
        return avg_loss
    
    def predict(self, users, items):
        self.model.eval()
        users = torch.tensor(users, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        with torch.no_grad():
            users = users.to(self.device)
            items = items.to(self.device)
            predictions =self.model(users, items)
        return predictions
    
    def predict_for_users(self, users, batch_size=1024):
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
        
        with torch.no_grad():
            # Process each user
            for user_id in users:
                user_tensor = torch.full((len(self.unique_items),), user_id, dtype=torch.long)
                
                all_scores = []
                for i in range(0, len(self.unique_items), batch_size):
                    batch_items = self.unique_items[i:i+batch_size]
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
    
    def get_top_k_recommendations(predictions, k=10, exclude_known=None):
        """
        Get top-K item recommendations for each user
        
        Args:
            predictions: Dictionary mapping user IDs to item scores
            k: Number of recommendations to return
            exclude_known: Dictionary mapping user IDs to sets of known item IDs to exclude
            
        Returns:
            Dictionary mapping user IDs to lists of recommended item IDs
        """
        recommendations = {}
        
        for user_id, scores in predictions.items():
            # Create a mask for items to exclude (optional)
            if exclude_known and user_id in exclude_known:
                mask = torch.ones_like(scores, dtype=torch.bool)
                for item_id in exclude_known[user_id]:
                    mask[item_id] = False
                masked_scores = scores.clone()
                masked_scores[~mask] = float('-inf')  # Set known items to negative infinity
            else:
                masked_scores = scores
            
            # Get top-K items
            values, indices = masked_scores.topk(k)
            recommendations[user_id] = indices.tolist()
            
        return recommendations
    
    def save(self, name='model.pth', path='models'):
        if not os.path.exists(path):
            os.makedirs(path, mode=777)
            
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime("_%Y-%m-%d_%H-%M-%S")
            
        name, ext = os.path.splitext(name)
        name += datetime_string + ext
        model_path = os.path.join(path, name)
        
        torch.save(self.model.state_dict(), model_path)
        
    def load():
        raise NotImplementedError