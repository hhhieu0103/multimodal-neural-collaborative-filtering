import numpy as np
import pandas as pd

from recom_ncf import NCFRecommender
import gc

class Evaluation:
    def __init__(
            self,
            recommender: NCFRecommender,
            test_data: pd.DataFrame,
            max_k=50,
    ):
        self.metrics = None
        self.recommender = recommender
        self.test_data = test_data
        self.max_k = max_k
        self.predictions = None
        self.ground_truth = None

        self.total_recommendation = pd.read_csv('C:/Users/Hieu/PycharmProjects/multimodal-neural-collaborative-filtering/data/metadata.csv')[['item_id', 'total_recommendations']]
        self.total_recommendation.set_index('item_id', inplace=True)

    def evaluate(self, k=10, user_batch_size=128, item_batch_size=1024):
        """
        Optimized evaluation method
        """

        if self.predictions is None or self.ground_truth is None:

            print("Creating ground truth sets...")
            self.ground_truth = {}
            for user, group in self.test_data.groupby('user_id'):
                self.ground_truth[user] = set(group['item_id'].values)

            print('Generating predictions...')
            test_unique_users = np.array(list(self.ground_truth.keys()))

            self.predictions = self.recommender.batch_predict_for_users(
                users=test_unique_users,
                k=self.max_k,
                user_batch_size=user_batch_size,
                item_batch_size=item_batch_size
            )

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

        pop_threshold = 1200
        pop_sum = 0
        pop_rate = 0

        total_users = len(self.ground_truth)
        processed_users = 0

        for user_id, true_items in self.ground_truth.items():
            if user_id not in self.predictions:
                continue

            # Get top-k recommendations for this user
            recommended_items = self.predictions[user_id][:k]
            recommended_items_pop = self.total_recommendation.loc[recommended_items, 'total_recommendations']
            pop_sum += np.sum(recommended_items_pop) / k
            pop_rate += (recommended_items_pop > pop_threshold).sum() / k

            # Calculate Hit Ratio - increment if any true item appears in recommendations
            if any(item in true_items for item in recommended_items):
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
            if processed_users % 10000 == 0 or processed_users == total_users:
                print(f"Processed {processed_users}/{total_users} users...")

        # Calculate final metrics
        hit_ratio = hits / total_users if total_users > 0 else 0
        ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        recall = np.mean(recalls) if recalls else 0
        arp = pop_sum / total_users if total_users > 0 else 0
        avg_pop_rate = pop_rate / total_users if total_users > 0 else 0

        self.metrics = {
            f'Hit Ratio@{k}': hit_ratio,
            f'NDCG@{k}': ndcg,
            f'Recall@{k}': recall,
            f'ARP@{k}': arp,
            f'Pop Ratio@{k}': avg_pop_rate,
        }

        gc.collect()

        return self.metrics