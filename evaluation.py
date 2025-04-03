import numpy as np
import pandas as pd
import torch
from recom_ncf import NCFRecommender
import gc
from time import time

class Evaluation:
    def __init__(
            self,
            recommender: NCFRecommender,
            test_data: pd.DataFrame,
            time_feature=None,
            df_metadata: pd.DataFrame=None,
            metadata_features=None,
            max_k=50
    ):
        self.metrics = None
        self.recommender = recommender
        self.test_data = test_data
        self.max_k = max_k
        self.predictions = None
        self.ground_truth = None

        self.time_feature = time_feature
        self.df_metadata = df_metadata
        self.metadata_features = metadata_features

    def predict_create_ground_truth(self, user_batch_size, item_batch_size):
        """
        Prepare evaluation data and generate predictions for all users in the test set.

        This method:
        1. Creates ground truth sets of relevant items for each user
        2. Extracts timestamps for each user (if time features are used)
        3. Analyzes metadata features (if metadata is used)
        4. Generates predictions for all users using the recommender model

        Args:
            user_batch_size: Number of users to process in each batch
            item_batch_size: Number of items to process in each batch
        """
        print("Starting evaluation preparation...")
        total_start = time()

        # Step 1: Create ground truth sets for all users
        self._create_ground_truth_sets()
        test_unique_users = list(self.ground_truth.keys())

        # Step 2: Extract timestamps for each user (if needed)
        timestamps = self._extract_user_timestamps(test_unique_users)

        # Step 3: Analyze metadata features (if needed)
        metadata_feature_dims = self._analyze_metadata_features()

        # Step 4: Generate predictions using the recommender model
        self._generate_predictions(
            test_unique_users,
            timestamps,
            user_batch_size,
            item_batch_size
        )

        # Log overall performance and clean up memory
        total_time = time() - total_start
        print(f"Evaluation preparation complete in {total_time:.2f} seconds!")

        self._cleanup_memory()

    def _create_ground_truth_sets(self):
        """Create sets of relevant items for each user in the test data."""
        print("Creating ground truth sets...")
        start_time = time()

        # Group by user_id and convert to a dictionary of sets
        self.ground_truth = {}
        for user, group in self.test_data.groupby('user_id'):
            self.ground_truth[user] = set(group['item_id'].values)

        num_users = len(self.ground_truth)
        avg_items = sum(len(items) for items in self.ground_truth.values()) / max(1, num_users)

        print(f"Ground truth created for {num_users} users with an average of {avg_items:.1f} items each")
        print(f"Ground truth creation completed in {time() - start_time:.2f} seconds")

    def _extract_user_timestamps(self, test_unique_users):
        """
        Extract the most recent timestamp for each user.

        Args:
            test_unique_users: List of unique user IDs

        Returns:
            List of timestamps (one per user) or None if time features aren't used
        """
        if self.time_feature is None:
            return None

        print("Extracting timestamps for each user...")
        start_time = time()

        # Use groupby to efficiently find maximum timestamp for each user
        latest_timestamps = self.test_data.groupby('user_id')[self.time_feature].max()

        # Create a list of timestamps in the same order as test_unique_users
        timestamps = [latest_timestamps.get(user, 0) for user in test_unique_users]

        print(f"Timestamp extraction completed in {time() - start_time:.2f} seconds")
        return timestamps

    def _analyze_metadata_features(self):
        """
        Analyze metadata features to determine their dimensions.

        Returns:
            List of feature dimensions or None if metadata isn't used
        """
        if self.df_metadata is None or self.metadata_features is None:
            return None

        print("Analyzing metadata features...")
        start_time = time()

        metadata_feature_dims = []
        high_dim_features = []

        for feature in self.metadata_features:
            if feature not in self.df_metadata.columns:
                print(f"Warning: Feature '{feature}' not found in metadata")
                continue

            # Find a non-null sample to determine feature dimension
            non_null_samples = self.df_metadata[~self.df_metadata[feature].isna()]

            if len(non_null_samples) == 0:
                print(f"Warning: Feature '{feature}' has no non-null values")
                continue

            # Get the first non-null sample
            sample = non_null_samples[feature].iloc[0]

            # Determine feature dimension
            if isinstance(sample, list):
                dim = len(sample)
                metadata_feature_dims.append(dim)

                # Track high-dimensional features
                if dim > 100:
                    high_dim_features.append((feature, dim))
            else:
                metadata_feature_dims.append(1)

        # Log information about feature dimensions
        print(f"Analyzed {len(metadata_feature_dims)} metadata features")

        if high_dim_features:
            print("High-dimensional features detected:")
            for feature, dim in high_dim_features:
                print(f"  - {feature}: {dim} dimensions")

        print(f"Metadata analysis completed in {time() - start_time:.2f} seconds")
        return metadata_feature_dims

    def _generate_predictions(self, test_unique_users, timestamps, user_batch_size, item_batch_size):
        """
        Generate predictions for all users using the recommender model.

        Args:
            test_unique_users: List of unique user IDs
            timestamps: List of timestamps (one per user) or None
            user_batch_size: Number of users to process in each batch
            item_batch_size: Number of items to process in each batch
        """
        print(f"Generating predictions for {len(test_unique_users)} users...")
        start_time = time()

        # Use the recommender's batch_predict_for_users method
        self.predictions = self.recommender.batch_predict_for_users(
            users=test_unique_users,
            timestamps=timestamps,
            df_metadata=self.df_metadata,
            metadata_features=self.metadata_features,
            k=self.max_k,
            user_batch_size=user_batch_size,
            item_batch_size=item_batch_size
        )

        prediction_time = time() - start_time

        # Calculate and log performance metrics
        predictions_per_second = len(test_unique_users) / prediction_time if prediction_time > 0 else 0

        print(f"Predictions generated for {len(test_unique_users)} users in {prediction_time:.2f} seconds")
        print(f"Prediction rate: {predictions_per_second:.1f} users/second")

    def _cleanup_memory(self):
        """Clean up memory after prediction."""
        import gc
        import torch

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self, k=10, user_batch_size=128, item_batch_size=1024):
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

        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions:
                continue

            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:k]

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