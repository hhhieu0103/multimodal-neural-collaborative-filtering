import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Splitter():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def leave_k_out_split(self, k_val=1, k_test=1, time_column='timestamp', rating_col='rating_imp'):
        """
        Split interactions dataframe using a leave-k-out strategy per user, ensuring test samples are positive.

        Args:
            k_val (int): Number of interactions to leave out per user for validation.
            k_test (int): Number of interactions to leave out per user for testing.
                          These will be selected from positive interactions only.
            time_column (str): Column name containing timestamps.
                               The most recent interactions will be used for testing/validation.
            rating_col (str): Column name containing the rating/preference (0 for dislike, 1 for like).

        Returns:
            tuple: (train_df, val_df, test_df) - The split dataframes
        """
        print(f"Splitting data with leave-{k_val + k_test}-out strategy ({k_val} for validation, {k_test} for testing)")
        print(f"Note: Ensuring test samples contain only positive interactions (where {rating_col} == 1)")

        # Copy the dataframe to avoid modifying the original
        df = self.df.copy()

        # Initialize empty dataframes for train, validation, and test sets
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Group interactions by user
        user_groups = df.groupby('user_id')

        # Calculate statistics for logging
        user_count = len(user_groups)
        interaction_counts = user_groups.size()
        min_interactions = interaction_counts.min()
        max_interactions = interaction_counts.max()
        avg_interactions = interaction_counts.mean()

        print(f"Total users: {user_count}")
        print(f"Interactions per user: min={min_interactions}, max={max_interactions}, avg={avg_interactions:.1f}")

        # Check if any users have too few interactions
        k_total = k_val + k_test
        users_with_insufficient_data = interaction_counts[interaction_counts < k_total].index.tolist()

        # Count users with insufficient positive interactions for testing
        users_with_insufficient_positives = 0

        # Track statistics for the split
        users_processed = 0
        train_interactions = 0
        val_interactions = 0
        test_interactions = 0

        # Process each user
        for user_id, user_df in tqdm(user_groups):
            users_processed += 1

            # Get number of interactions for this user
            n_interactions = len(user_df)

            # If user has too few interactions, put all in training set
            if n_interactions < k_total:
                train_dfs.append(user_df)
                train_interactions += n_interactions
                continue

            # Sort by timestamp (most recent last)
            if time_column in user_df.columns:
                user_df = user_df.sort_values(by=time_column)
            else:
                print(f"Warning: Time column '{time_column}' not found. Using original order.")

            # Check if user has enough positive interactions for testing
            positive_df = user_df[user_df[rating_col] == 1]
            n_positives = len(positive_df)

            if n_positives < k_test:
                # Not enough positive interactions for testing
                users_with_insufficient_positives += 1

                # Add everything to training except for validation
                val_df = user_df.tail(k_val)
                train_df = user_df.iloc[:-k_val] if k_val > 0 else user_df
                test_df = pd.DataFrame(columns=user_df.columns)  # Empty test set
            else:
                # User has enough positive interactions for testing

                # Get the k_test most recent positive interactions for test set
                positive_df = positive_df.sort_values(by=time_column)
                test_df = positive_df.tail(k_test)

                # Remove the test samples from the user dataframe
                test_indices = test_df.index
                remaining_df = user_df[~user_df.index.isin(test_indices)]

                # Select validation set from remaining data
                if k_val > 0:
                    # Try to get the most recent interactions for validation
                    val_df = remaining_df.sort_values(by=time_column).tail(k_val)
                    # Remove validation samples and keep the rest for training
                    val_indices = val_df.index
                    train_df = remaining_df[~remaining_df.index.isin(val_indices)]
                else:
                    val_df = pd.DataFrame(columns=user_df.columns)
                    train_df = remaining_df

            # Add to the result dataframes
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)

            # Update statistics
            train_interactions += len(train_df)
            val_interactions += len(val_df)
            test_interactions += len(test_df)

        # Combine all user dataframes
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Print statistics about the split
        total_interactions = train_interactions + val_interactions + test_interactions
        print(f"{users_with_insufficient_positives} users had insufficient positive interactions for testing.")
        print(f"Split complete: {total_interactions} total interactions")
        print(f"Train set: {train_interactions} interactions ({train_interactions / total_interactions * 100:.1f}%)")
        print(f"Validation set: {val_interactions} interactions ({val_interactions / total_interactions * 100:.1f}%)")
        print(f"Test set: {test_interactions} interactions ({test_interactions / total_interactions * 100:.1f}%)")

        # Verify test set has only positive interactions
        if len(test_df) > 0:
            positive_test_ratio = (test_df[rating_col] == 1).mean() * 100
            print(f"Test set positive ratio: {positive_test_ratio:.1f}% (should be 100%)")

        return train_df, val_df, test_df

    def random_split(self, test_size=0.1, random_state=42):
        df_train_val, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        df_train, df_val = train_test_split(df_train_val, test_size=(test_size / (1-test_size)), random_state=random_state)

        return df_train, df_val, df_test