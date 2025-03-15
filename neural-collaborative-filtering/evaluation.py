import numpy as np
import torch

class Evaluation():
    def __init__(self, recommender=None, test_data=None, k=10, predictions=None, ground_truth=None, user_batch_size=1024, item_batch_size=8192):
        self.recommender = recommender
        self.test_data = test_data
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.k = k
        self.user_batch_size = user_batch_size
        self.item_batch_size = item_batch_size
        
    def predict_create_ground_truth(self):
        import gc  # For garbage collection
        
        self.predictions = {}
        self.ground_truth = {}
        test_unique_users = self.test_data['user_idx'].unique()
        
        # Create ground truth efficiently using NumPy arrays
        user_array = self.test_data['user_idx'].values
        item_array = self.test_data['item_idx'].values
        
        # Find unique users and create ground truth sets
        unique_users = np.unique(user_array)
        for user in unique_users:
            self.ground_truth[user] = set(item_array[user_array == user])
        
        # Get predictions for all users at once
        # If your GPU can handle it, this will be much faster
        all_predictions = self.batch_predict_for_users(test_unique_users)
        
        # Process each user's predictions
        for user_id, scores in all_predictions.items():
            if isinstance(scores, torch.Tensor):
                scores_np = scores.numpy()
            else:
                scores_np = np.array(scores)
                
            # Use argpartition for efficient top-k selection
            top_k_indices = np.argpartition(scores_np, -self.k)[-self.k:]
            # Sort just the top k items by score
            top_k_indices = top_k_indices[np.argsort(-scores_np[top_k_indices])]
            self.predictions[user_id] = top_k_indices.tolist()
        
        # Clean up
        del all_predictions
        gc.collect()
        torch.cuda.empty_cache()
    
    def batch_predict_for_users(self, users, items=None):
        """
        Generate predictions for all items for all users at once
        
        Args:
            users: List or array of user IDs
            items: Optional tensor of item IDs. If None, all items are used.
            
        Returns:
            Dictionary mapping user IDs to tensors of scores for all items
        """
        import gc  # For garbage collection
        
        self.recommender.model.eval()
        predictions = {}
        
        # Convert users to numpy for final processing
        if isinstance(users, torch.Tensor):
            users_np = users.cpu().numpy()
        else:
            users_np = np.array(users)
        
        # Setup item tensor on device
        if items is None:
            items = torch.arange(len(self.recommender.unique_items), device=self.recommender.device)
        elif not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long, device=self.recommender.device)
        
        num_users = len(users_np)
        num_items = len(items)
        
        with torch.no_grad():
            # Process users in batches
            for i in range(0, num_users, self.user_batch_size):
                print(f'Processing user batch {i+1}/{num_users}')
                batch_users_np = users_np[i:i+self.user_batch_size]
                batch_size = len(batch_users_np)
                users_tensor = torch.tensor(batch_users_np, dtype=torch.long, device=self.recommender.device)
                
                # Allocate scores tensor for this batch
                all_scores_gpu = torch.zeros((batch_size, num_items), device=self.recommender.device)
                
                # Process items in batches
                for j in range(0, num_items, self.item_batch_size):
                    batch_items = items[j:j+self.item_batch_size]
                    item_batch_size = len(batch_items)
                    
                    # Create user-item pairs
                    users_matrix = users_tensor.repeat_interleave(item_batch_size)
                    items_matrix = batch_items.repeat(batch_size)
                    
                    # Make predictions
                    batch_scores = self.recommender.model(users_matrix, items_matrix)
                    
                    # Reshape and store scores
                    batch_scores = batch_scores.view(batch_size, item_batch_size)
                    all_scores_gpu[:, j:j+item_batch_size] = batch_scores
                    
                    # Clean up intermediate tensors
                    del users_matrix, items_matrix, batch_scores
                
                # Store predictions for this batch of users
                for idx, user_id in enumerate(batch_users_np):
                    # Get the user's scores and convert to numpy right away
                    user_scores = all_scores_gpu[idx].cpu().numpy()
                    
                    # Find top-k items
                    top_k_indices = np.argpartition(user_scores, -self.k)[-self.k:]
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

    def evaluate(self):
        """
        Evaluate the model using the specified metrics.
        
        Returns:
            Dictionary containing metric names and scores
        """
        
        if self.predictions is None or self.ground_truth is None:
            self.predict_create_ground_truth()
        
        self.metrics = {
            'Hit Ratio@{}'.format(self.k): self.hit_ratio_at_k(),
            'NDCG@{}'.format(self.k): self.ndcg_at_k(),
            'Recall@{}'.format(self.k): self.recall_at_k()
        }
        
        return self.metrics
    
    def hit_ratio_at_k(self):
        """
        Calculate Hit Ratio@k for recommendations.
        
        Returns:
            Hit Ratio@k score (between 0 and 1)
        """
        hits = 0
        total_users = len(self.ground_truth)
        
        for user_idx, true_items in self.ground_truth.items():
            if user_idx not in self.predictions:
                continue
                
            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_idx][:self.k]
            
            # Check if any true items appear in the recommendations
            if any(item in true_items for item in recommended_items):
                hits += 1

        return hits / total_users if total_users > 0 else 0
    
    def ndcg_at_k(self):
        """
        Calculate NDCG@k for recommendations.
        
        Returns:
            NDCG@k score (between 0 and 1)
        """
        ndcg_scores = []
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions or not true_items:
                continue
                
            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:self.k]
            
            # Create a binary relevance vector
            relevance = [1 if item in true_items else 0 for item in recommended_items]
            
            # Calculate DCG using vectorized operations
            discounts = np.log2(np.arange(2, len(relevance) + 2))
            dcg = np.sum(relevance / discounts)
            
            # Calculate ideal DCG (IDCG)
            ideal_relevance = [1] * min(len(true_items), self.k)
            idcg = np.sum(ideal_relevance / discounts[:len(ideal_relevance)])
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    def recall_at_k(self):
        """
        Calculate Recall@k for recommendations.
        
        Returns:
            Recall@k score (between 0 and 1)
        """
        recalls = []
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions or not true_items:
                continue
                
            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:self.k]
            
            # Count relevant items in the recommendations
            relevant_set = set(true_items)
            num_relevant = sum(1 for item in recommended_items if item in relevant_set)
            
            # Calculate recall for this user
            user_recall = num_relevant / len(true_items)
            recalls.append(user_recall)
        
        return np.mean(recalls) if recalls else 0