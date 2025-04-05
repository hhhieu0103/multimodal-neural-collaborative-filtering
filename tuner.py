import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import itertools
import json
import os
from datetime import datetime

from dataset import NCFDataset
from recom_ncf import NCFRecommender
from evaluation import Evaluation

class NCFTuner:
    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        unique_users,
        unique_items,
        time_feature=None,
        metadata=None,
        metadata_features=None,
        k_values=[50, 20, 10],
        results_dir='tuning_results'
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.metadata = metadata
        self.unique_users = unique_users
        self.unique_items = unique_items
        self.results_dir = results_dir
        self.k_values = k_values
        self.time_feature = time_feature
        self.metadata_features = metadata_features
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.use_time = time_feature is not None
        self.use_metadata = metadata is not None and metadata_features is not None and len(metadata_features) > 0
        
        # Default parameter grid
        self.param_grid = {
            'factors': [8, 16, 32, 64],
            'mlp_user_item_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'epochs': [1],
            'optimizer': ['sgd', 'adam', 'adagrad'],
            'dropout': [0.0, 0.2, 0.5],
            'weight_decay': [0.0, 0.0001, 0.00001],
            'loss_fn': ['bce', 'mse', 'bpr'],
            'batch_size': [128, 256, 512],
        }

        if self.use_time:
            self.param_grid['mlp_time_dim'] = [4, 8, 16]

        if self.use_metadata:
            self.param_grid['mlp_metadata_embedding_dims'] = [[8, 8, 8], [16, 16, 16], [32, 32, 32]]


    def set_param_grid(self, param_grid):
        """Set the parameter grid to use"""
        self.param_grid = param_grid

    def update_param_grid(self, param_grid):
        """Update the parameter grid with custom values"""
        self.param_grid.update(param_grid)
        
    def run_experiment(self, params):
        """Run a single experiment with the given parameters"""
        print(f"Running experiment with params: {params}")

        dataset_params = {}

        if self.use_time:
            dataset_params['time_feature'] = self.time_feature

        if self.use_metadata:
            dataset_params['df_metadata'] = self.metadata
            dataset_params['metadata_features'] = self.metadata_features

        # Create data loaders
        train_dataset = NCFDataset(df_interaction=self.train_data, **dataset_params)
        val_dataset = NCFDataset(df_interaction=self.test_data, **dataset_params)

        dataloader_params = {
            'batch_size': params['batch_size'],
            'shuffle': True,
            'num_workers': 4,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'pin_memory': True
        }
        
        train_loader = DataLoader(train_dataset, **dataloader_params)
        val_loader = DataLoader(val_dataset, **dataloader_params)

        recommender_params = {
            'unique_users': self.unique_users,
            'unique_items': self.unique_items,
            'factors': params['factors'],
            'mlp_user_item_dim': params['mlp_user_item_dim'],
            'learning_rate': params['learning_rate'],
            'epochs': params['epochs'],
            'optimizer': params['optimizer']
        }

        if self.use_time:
            recommender_params['mlp_time_dim'] = params['mlp_time_dim']

        if self.use_metadata:
            recommender_params['mlp_metadata_embedding_dims'] = params['mlp_metadata_embedding_dims']
            recommender_params['mlp_metadata_feature_dims'] = train_dataset.get_feature_dims()

            # Create model with the specified parameters
        model = NCFRecommender(**recommender_params)
        
        # Train the model
        model.fit(train_loader, val_loader)
        
        # Evaluate the model at different k values
        eval_results = {}
        evaluator_params = {
            'recommender': model,
            'test_data': self.test_data,
            'max_k': max(self.k_values),
        }

        if self.use_time:
            evaluator_params['time_feature'] = self.time_feature

        if self.use_metadata:
            evaluator_params['df_metadata'] = self.metadata
            evaluator_params['metadata_features'] = self.metadata_features

        evaluator = Evaluation(**evaluator_params)
        for k in self.k_values:
            metrics = evaluator.evaluate(k)
            eval_results[f'k={k}'] = metrics
            
        print(eval_results)
        
        # Save the training losses
        eval_results['train_loss'] = model.train_losses
        eval_results['val_loss'] = model.val_losses
        
        return eval_results
    
    def perform_grid_search(self, num_combinations=None):
        """
        Perform grid search over parameter combinations
        
        Args:
            num_combinations: If provided, randomly sample this many combinations
                             instead of trying all combinations
        """
        # Generate all parameter combinations
        keys = self.param_grid.keys()
        combinations = list(itertools.product(*(self.param_grid[key] for key in keys)))
        
        # Sample a subset if specified
        if num_combinations and num_combinations < len(combinations):
            combinations = np.random.choice(combinations, num_combinations, replace=False)
        
        results = []
        
        # Run each combination
        for combo_idx, combo_values in enumerate(combinations):
            params = {key: value for key, value in zip(keys, combo_values)}
            
            # Run the experiment
            experiment_results = self.run_experiment(params)
            
            # Save this experiment
            result_entry = {
                'params': params,
                'metrics': experiment_results
            }
            results.append(result_entry)
            
            # Save intermediate results
            self._save_results(results, f"intermediate_{combo_idx}")
            
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_results(results, f"final_results_{timestamp}")
        
        return results
    
    def perform_random_search(self, num_trials=10, prevent_duplicates=True):
        """
        Perform random search with the specified number of trials
        
        Args:
            num_trials: Number of parameter combinations to try
            prevent_duplicates: If True, ensures no parameter set is tried twice
        """
        results = []
        tried_configs = set()  # Keep track of configs we've already tried
        
        trial = 0
        max_attempts = num_trials * 10  # Set a maximum to prevent infinite loops
        attempts = 0
        
        while trial < num_trials and attempts < max_attempts:
            attempts += 1
            
            # Sample random parameters
            params = {}
            for key, values in self.param_grid.items():
                if len(values) == 0:
                    continue
                
                idx = np.random.randint(0, len(values))
                params[key] = values[idx]  
        
            params_hashable = tuple(
                (k, tuple(v)) if isinstance(v, list) else (k, v)
                for k, v in sorted(params.items())
            )
            
            if prevent_duplicates and params_hashable in tried_configs:
                continue
                
            # Mark this configuration as tried
            tried_configs.add(params_hashable)
            
            # Run the experiment
            experiment_results = self.run_experiment(params)
            
            # Save this experiment
            result_entry = {
                'params': params,
                'metrics': experiment_results
            }
            results.append(result_entry)
            
            # Save intermediate results
            self._save_results(results, f"random_search_intermediate_{trial}")
            
            trial += 1
        
        if trial < num_trials:
            print(f"Warning: Could only find {trial} unique parameter combinations out of requested {num_trials}")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self._save_results(results, f"random_search_final_{timestamp}")
        
        return results, result_path
    
    def _save_results(self, results, filename):
        """Save results to a file"""
        filepath = os.path.join(self.results_dir, f"{filename}.json")
        
        # Convert any non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
        
        # Apply conversion recursively
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        return filepath

    def __load_results(self, filepath):
        """Load results from a file"""
        if filepath:
            with open(filepath, 'r') as f:
                results = json.load(f)
        else:
            # Find the most recent final results file
            files = [f for f in os.listdir(self.results_dir) if f.startswith('random_search_final_') and f.endswith('.json')]
            if not files:
                raise ValueError("No results file found")

            latest_file = max(files)
            with open(os.path.join(self.results_dir, latest_file), 'r') as f:
                results = json.load(f)

        return results

    def analyze_results(self, results_file=None):
        """Analyze tuning results and return best parameters"""
        results = self.__load_results(results_file)
        
        # Analyze for different metrics
        best_params = {}
        metrics_to_analyze = ['Hit Ratio@10', 'NDCG@10', 'Recall@10']
        
        for metric in metrics_to_analyze:
            sorted_results = sorted(
                results, 
                key=lambda x: x['metrics']['k=10'][metric], 
                reverse=True
            )
            
            best_params[metric] = {
                'params': sorted_results[0]['params'],
                'value': sorted_results[0]['metrics']['k=10'][metric]
            }
            
        return best_params
        
    def plot_results(self, results_file=None):
        """Plot the results of hyperparameter tuning"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn are required for plotting")
            return
            
        results = self.__load_results(results_file)
                
        # Convert results to DataFrame for easier analysis
        rows = []
        for experiment in results:
            params = experiment['params']
            metrics = experiment['metrics']['k=10']  # Use k=10 metrics
            
            row = {**params, **metrics}
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Plot heatmaps for parameters and metrics
        plt.figure(figsize=(15, 10))
        
        for i, param in enumerate(self.param_grid.keys()):
            for j, metric in enumerate(['Hit Ratio@10', 'NDCG@10', 'Recall@10']):
                plt.subplot(len(self.param_grid.keys()), 3, i*3 + j + 1)
                pivot = df.pivot_table(
                    index=param,
                    values=metric,
                    aggfunc='mean'
                )
                sns.heatmap(pivot, annot=True, cmap='viridis')
                plt.title(f"{metric} by {param}")
                
        plt.tight_layout()
        
        # Plot learning curves for best model
        best_idx = df['NDCG@10'].idxmax()
        best_params = {k: df.iloc[best_idx][k] for k in self.param_grid.keys()}
        
        # Find the experiment with these parameters
        best_experiment = next(
            exp for exp in results 
            if all(exp['params'][k] == best_params[k] for k in best_params)
        )
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(best_experiment['metrics']['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(best_experiment['metrics']['val_loss'])
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
        
        return df

# Example usage:
"""
# 1. Load your data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 2. Get unique users and items
unique_users = train_data['user_idx'].unique()
unique_items = train_data['item_idx'].unique()

# 3. Create the tuner
tuner = NCFTuner(
    train_data=train_data,
    test_data=test_data,
    unique_users=unique_users,
    unique_items=unique_items
)

# 4. Customize parameter grid if needed
tuner.update_param_grid({
    'factors': [16, 32, 64],
    'epochs': [20, 30]
})

# 5. Run random search (faster than grid search)
results = tuner.perform_random_search(num_trials=10)

# 6. Or run grid search (more comprehensive)
# results = tuner.perform_grid_search(num_combinations=15)

# 7. Analyze results
best_params = tuner.analyze_results()
print("Best parameters:", best_params)

# 8. Plot results
tuner.plot_results()
"""