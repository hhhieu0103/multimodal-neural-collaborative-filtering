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
        
        print("Starting evaluation preparation...")
        self.predictions = {}
        self.ground_truth = {}
        test_unique_users = self.test_data['user_idx'].unique()
        
        print(f"Creating ground truth sets for {len(test_unique_users)} users...")
        # Create ground truth efficiently using NumPy arrays
        user_array = self.test_data['user_idx'].values
        item_array = self.test_data['item_idx'].values
        
        # Find unique users and create ground truth sets
        unique_users = np.unique(user_array)
        user_count = len(unique_users)
        
        for i, user in enumerate(unique_users):
            self.ground_truth[user] = set(item_array[user_array == user])
        
        print("Generating predictions...")
        # Get predictions for all users - this now returns only the top-K indices directly
        self.predictions = self.recommender.batch_predict_for_users(
            test_unique_users,
            user_batch_size=self.user_batch_size,
            item_batch_size=self.item_batch_size,
            k=self.k
        )
        
        # Since the recommender now returns the top-K indices directly, we don't need to process them further
        print("Evaluation preparation complete!")
        gc.collect()
        torch.cuda.empty_cache()
    
    def evaluate(self):
        """
        Evaluate the model using the specified metrics.
        
        Returns:
            Dictionary containing metric names and scores
        """
        import gc
        
        if self.predictions is None or self.ground_truth is None:
            self.predict_create_ground_truth()
        
        print("Calculating Hit Ratio...")
        hit_ratio = self.hit_ratio_at_k()
        gc.collect()
        
        print("Calculating NDCG...")
        ndcg = self.ndcg_at_k()
        gc.collect()
        
        print("Calculating Recall...")
        recall = self.recall_at_k()
        gc.collect()
        
        self.metrics = {
            'Hit Ratio@{}'.format(self.k): hit_ratio,
            'NDCG@{}'.format(self.k): ndcg,
            'Recall@{}'.format(self.k): recall
        }
        
        print("Evaluation complete!")
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