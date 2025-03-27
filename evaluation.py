import numpy as np
import pandas as pd
import torch
from recom_ncf import NCFRecommender

class Evaluation():
    def __init__(
        self,
        recommender: NCFRecommender,
        test_data: pd.DataFrame,
        df_metadata: pd.DataFrame,
        time_feature='timestamp',
        metadata_features=[],
        max_k=50
    ):
        self.recommender = recommender
        self.test_data = test_data
        self.max_k = max_k
        self.time_feature = time_feature
        self.predictions = None
        self.ground_truth = None
        
        self.df_metadata = df_metadata
        self.metadata_features = metadata_features
        
    def predict_create_ground_truth(self, user_batch_size, item_batch_size):
        import gc  # For garbage collection
        
        print("Starting evaluation preparation...")
        self.predictions = {}
        self.ground_truth = {}
        test_unique_users = self.test_data['user_idx'].unique()
        
        print(f"Creating ground truth sets for {len(test_unique_users)} users...")
        # Create ground truth efficiently using NumPy arrays
        user_array = self.test_data['user_idx'].values
        
        # Find unique users and create ground truth sets
        unique_users = np.unique(user_array)
        
        self.ground_truth = self.test_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        print("Getting most recent timestamps for each user...")        
        timestamp_array = self.test_data[self.time_feature].values    
        latest_timestamps = {}
        for i in range(len(user_array)):
            user = user_array[i]
            timestamp = timestamp_array[i]
            if user not in latest_timestamps or timestamp > latest_timestamps[user]:
                latest_timestamps[user] = timestamp
        timestamps = [latest_timestamps.get(user, 0) for user in unique_users]
        
        print("Generating predictions...")
        # Get predictions for all users - this now returns only the top-K indices directly
        # self.predictions = self.recommender.batch_predict_for_users(test_unique_users, timestamps, df_metadata=self.df_metadata, metadata_features=self.metadata_features, k=self.max_k, user_batch_size=user_batch_size, item_batch_size=item_batch_size)
        self.predictions = self.recommender.batch_predict_for_users(test_unique_users, timestamps, df_metadata=self.df_metadata, metadata_features=self.metadata_features, k=self.max_k)
        
        
        # Since the recommender now returns the top-K indices directly, we don't need to process them further
        print("Evaluation preparation complete!")
        gc.collect()
        torch.cuda.empty_cache()
    
    def evaluate(self, k=10, user_batch_size=512, item_batch_size=4096):
        """
        Evaluate the model using the specified metrics.
        
        Returns:
            Dictionary containing metric names and scores
        """
        import gc
        
        if self.predictions is None or self.ground_truth is None:
            self.predict_create_ground_truth(user_batch_size=user_batch_size, item_batch_size=item_batch_size)
            
        if k > self.max_k:
            print(f'The predictions were made with k={self.max_k}. The evaluation metrics can only be calculated with k <= {self.max_k}. The k parameter is {k}, which exceeds the max k value. Falling back to k={self.max_k}')
            k = self.max_k
        
        print("Calculating Hit Ratio...")
        hit_ratio = self.hit_ratio_at_k(k)
        gc.collect()
        
        print("Calculating NDCG...")
        ndcg = self.ndcg_at_k(k)
        gc.collect()
        
        print("Calculating Recall...")
        recall = self.recall_at_k(k)
        gc.collect()
        
        self.metrics = {
            'Hit Ratio@{}'.format(k): hit_ratio,
            'NDCG@{}'.format(k): ndcg,
            'Recall@{}'.format(k): recall
        }
        
        print("Evaluation complete!")
        return self.metrics
    
    def hit_ratio_at_k(self, k):
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
            recommended_items = self.predictions[user_idx][:k]
            
            # Check if any true items appear in the recommendations
            if any(item in true_items for item in recommended_items):
                hits += 1

        return hits / total_users if total_users > 0 else 0
    
    def ndcg_at_k(self, k):
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
            recommended_items = self.predictions[user_id][:k]
            
            # Create a binary relevance vector
            relevance = [1 if item in true_items else 0 for item in recommended_items]
            
            # Calculate DCG using vectorized operations
            discounts = np.log2(np.arange(2, len(relevance) + 2))
            dcg = np.sum(relevance / discounts)
            
            # Calculate ideal DCG (IDCG)
            ideal_relevance = [1] * min(len(true_items), k)
            idcg = np.sum(ideal_relevance / discounts[:len(ideal_relevance)])
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    def recall_at_k(self, k):
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
            recommended_items = self.predictions[user_id][:k]
            
            # Count relevant items in the recommendations
            relevant_set = set(true_items)
            num_relevant = sum(1 for item in recommended_items if item in relevant_set)
            
            # Calculate recall for this user
            user_recall = num_relevant / len(true_items)
            recalls.append(user_recall)
        
        return np.mean(recalls) if recalls else 0