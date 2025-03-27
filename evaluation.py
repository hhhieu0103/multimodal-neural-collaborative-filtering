import numpy as np
import pandas as pd
import torch
from recom_ncf import NCFRecommender

class Evaluation():
    def __init__(
            self,
            recommender: NCFRecommender,
            test_data: pd.DataFrame,
            df_metadata: pd.DataFrame = None,
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
        """
        Optimized method to create predictions and ground truth, keeping dynamic batch size adjustment
        """
        import gc
        from time import time

        print("Starting evaluation preparation...")

        # Create ground truth sets once (this is fast)
        print("Creating ground truth sets...")
        self.ground_truth = self.test_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
        test_unique_users = list(self.ground_truth.keys())

        print("Getting most recent timestamps for each user...")
        latest_timestamps = {}
        for _, row in self.test_data.iterrows():
            user = row['user_idx']
            timestamp = row[self.time_feature]
            if user not in latest_timestamps or timestamp > latest_timestamps[user]:
                latest_timestamps[user] = timestamp

        timestamps = [latest_timestamps.get(user, 0) for user in test_unique_users]

        print("Generating predictions...")
        start_time = time()

        # Use the recommender's batch_predict_for_users method with dynamic batch sizing
        # This preserves the original memory management approach
        self.predictions = self.recommender.batch_predict_for_users(
            test_unique_users,
            timestamps,
            df_metadata=self.df_metadata,
            metadata_features=self.metadata_features,
            k=self.max_k,
            user_batch_size=user_batch_size,
            item_batch_size=item_batch_size
        )

        print(f"Predictions generated in {time() - start_time:.2f} seconds!")
        print("Evaluation preparation complete!")

        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, k=10, user_batch_size=512, item_batch_size=4096):
        """
        Optimized evaluation method
        """
        import gc
        from time import time

        start_time = time()
        if self.predictions is None or self.ground_truth is None:
            self.predict_create_ground_truth(user_batch_size=user_batch_size, item_batch_size=item_batch_size)

        if k > self.max_k:
            print(f"Warning: k={k} exceeds max_k={self.max_k}. Using k={self.max_k} instead.")
            k = self.max_k

        # Calculate all metrics in one pass through the data for efficiency
        print("Calculating metrics...")
        hits = 0
        ndcg_scores = []
        recalls = []

        # Precompute discount factors for NDCG
        discounts = 1.0 / np.log2(np.arange(2, k + 2))

        total_users = len(self.ground_truth)
        processed_users = 0

        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions:
                continue

            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:k]

            # Calculate Hit Ratio - increment if any true item appears in recommendations
            hit = 0
            if any(item in true_items for item in recommended_items):
                hit = 1
                hits += 1

            # Calculate NDCG
            relevance = np.array([1 if item in true_items else 0 for item in recommended_items])
            dcg = np.sum(relevance * discounts)

            # Calculate ideal DCG
            ideal_relevance = np.ones(min(len(true_items), k))
            idcg = np.sum(ideal_relevance * discounts[:len(ideal_relevance)])

            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

            # Calculate Recall
            num_relevant = np.sum(relevance)
            recall = num_relevant / len(true_items) if true_items else 0
            recalls.append(recall)

            # Show progress periodically
            processed_users += 1
            if processed_users % 100 == 0 or processed_users == total_users:
                print(f"Processed {processed_users}/{total_users} users...")

        # Calculate final metrics
        hit_ratio = hits / total_users if total_users > 0 else 0
        ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        recall = np.mean(recalls) if recalls else 0

        self.metrics = {
            f'Hit Ratio@{k}': hit_ratio,
            f'NDCG@{k}': ndcg,
            f'Recall@{k}': recall
        }

        # Clear memory
        gc.collect()

        print(f"Evaluation complete in {time() - start_time:.2f} seconds!")
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