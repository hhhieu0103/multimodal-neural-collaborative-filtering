import numpy as np
import heapq

class Evaluation():
    def __init__(self, recommender=None, test_data=None, k=10, predictions=None, ground_truth=None):
        self.recommender = recommender
        self.test_data = test_data
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.k = k
        
    def predict_create_ground_truth(self):
        # Generate predictions and prepare ground truth
        self.predictions = {}  # user_id -> ordered list of recommended items
        self.ground_truth = {}  # user_id -> set of relevant items from test set
        test_unique_users = self.test_data['user_idx'].unique()

        # For each user in test set
        for user_idx in test_unique_users:
            relevant_items = set(self.test_data[self.test_data['user_idx'] == user_idx]['item_idx'])
            self.ground_truth[user_idx] = relevant_items
            
            scores = self.recommender.predict_for_users([user_idx])[user_idx].numpy()
            top_k_items = []
            
            for item_id, score in enumerate(scores):
                # For a min-heap, we use negative score to get highest scores
                if len(top_k_items) < self.k:
                    heapq.heappush(top_k_items, (score, item_id))
                elif score > top_k_items[0][0]:
                    heapq.heappushpop(top_k_items, (score, item_id))
                    
            # Convert heap to sorted list (highest scores first)
            sorted_items = [item_id for score, item_id in sorted(top_k_items, reverse=True)]
            
            self.predictions[user_idx] = sorted_items
        
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
        
        Args:
            predictions: Dictionary mapping user IDs to ordered lists of recommended item IDs
            ground_truth: Dictionary mapping user IDs to sets of relevant item IDs
            k: Number of top recommendations to consider
        
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
        
        Args:
            predictions: Dictionary mapping user IDs to ordered lists of recommended item IDs
            ground_truth: Dictionary mapping user IDs to sets of relevant item IDs
            k: Number of top recommendations to consider
        
        Returns:
            NDCG@k score (between 0 and 1)
        """
        ndcg_scores = []
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions or not true_items:
                continue
                
            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:self.k]
            
            # Calculate DCG
            dcg = 0
            for i, item in enumerate(recommended_items):
                if item in true_items:
                    # Using binary relevance (1 if relevant, 0 if not)
                    # Position i+1 because we start counting from 0
                    dcg += 1 / np.log2(i + 2)  # log base 2 of position + 1
            
            # Calculate ideal DCG (IDCG)
            # IDCG is the DCG value if recommendations were perfect
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), self.k)))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    
    def recall_at_k(self):
        """
        Calculate Recall@k for recommendations.
        
        Args:
            predictions: Dictionary mapping user IDs to ordered lists of recommended item IDs
            ground_truth: Dictionary mapping user IDs to sets of relevant item IDs
            k: Number of top recommendations to consider
        
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
            num_relevant = sum(1 for item in recommended_items if item in true_items)
            
            # Calculate recall for this user
            user_recall = num_relevant / len(true_items)
            recalls.append(user_recall)
        
        return sum(recalls) / len(recalls) if recalls else 0