import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import NCFDataset
from recom_ncf import NCFRecommender
from evaluation import Evaluation
from ncf import ModelType

from helpers.mem_map_dataloader import MemMapDataLoader
from helpers.dataloader_custom_functions import collate_fn, worker_init_fn


def _dict_to_hashable(dictionary):
    hashable = []
    for k, v in sorted(dictionary.items()):
        if isinstance(v, list):
            hashable.append((k, tuple(v)))
        elif isinstance(v, dict):
            hashable.append((k, tuple(v.items())))
        else:
            hashable.append((k, v))
    return tuple(hashable)


def _extract_configs(results):
    """Extract parameter configurations from results"""
    configs = set()
    for result in results:
        config = _dict_to_hashable(result['params'])
        configs.add(config)
    return configs

class NCFTuner:
    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        unique_users,
        unique_items,
        feature_dims=None, # Dictionary where keys are feature names, values are lists of output dimensions
        df_features=None,
        k_values=[50, 20, 10],
        results_dir='tuning_results',
        image_dataloader: MemMapDataLoader = None,
        model_type: ModelType = ModelType.EARLY_FUSION
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.unique_users = unique_users
        self.unique_items = unique_items
        self.results_dir = results_dir
        self.k_values = k_values
        self.df_features = df_features
        self.feature_dims = feature_dims
        self.image_dataloader = image_dataloader
        self.model_type = model_type
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Default parameter grid
        self.param_grid = {
            'factors': [8, 16, 32, 64],
            'mlp_user_item_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'epochs': [100],
            'optimizer': ['sgd', 'adam', 'adagrad'],
            'dropout': [0.0, 0.2, 0.5],
            'weight_decay': [0.0, 0.0001, 0.00001],
            'loss_fn': ['bce', 'mse', 'bpr'],
            'batch_size': [128, 256, 512],
        }

    def set_param_grid(self, param_grid):
        """Set the parameter grid to use"""
        self.param_grid = param_grid

    def update_param_grid(self, param_grid):
        """Update the parameter grid with custom values"""
        self.param_grid.update(param_grid)
        
    def run_experiment(self, params):
        """Run a single experiment with the given parameters"""
        print(f"Running experiment with params: {params}")
        dataset_params = {
            'df_features': self.df_features,
            'image_dataloader': self.image_dataloader,
            'feature_dims': params.get('mlp_feature_dims', None),
        }

        # Create data loaders
        train_dataset = NCFDataset(df_interaction=self.train_data, **dataset_params)
        val_dataset = NCFDataset(df_interaction=self.val_data, **dataset_params)

        dataloader_params = {
            'batch_size': params['batch_size'],
            'num_workers': 4,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'pin_memory': True,
            'pin_memory_device': 'cuda',
            'collate_fn': collate_fn,
        }
        
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

        recommender_params = {
            'unique_users': self.unique_users,
            'unique_items': self.unique_items,
            'factors': params['factors'],
            'mlp_user_item_dim': params['mlp_user_item_dim'],
            'learning_rate': params['learning_rate'],
            'epochs': params['epochs'],
            'optimizer': params['optimizer'],
            'dropout': params['dropout'],
            'weight_decay': params['weight_decay'],
            'loss_fn': params['loss_fn'],
            'mlp_feature_dims': params.get('mlp_feature_dims', None),
            'image_dataloader': self.image_dataloader,
            'image_dim': params.get('image_dim', None),
            'df_features': self.df_features,
            'model_type': self.model_type,
        }

        # Create model with the specified parameters
        model = NCFRecommender(**recommender_params)
        
        # Train the model
        model.fit(train_data=train_loader, val_data=val_loader)
        
        # Evaluate the model at different k values
        eval_results = {}
        evaluator_params = {
            'recommender': model,
            'test_data': self.test_data,
            'max_k': max(self.k_values),
        }

        if self.image_dataloader is not None:
            self.image_dataloader.open_lmdb()

        evaluator = Evaluation(**evaluator_params)
        for k in self.k_values:
            metrics = evaluator.evaluate(k)
            eval_results[f'k={k}'] = metrics
            
        print(eval_results)

        # Save the training losses
        eval_results['train_loss'] = model.train_losses
        eval_results['val_loss'] = model.val_losses
        
        return eval_results

    def perform_random_search(self, num_trials=10, prevent_duplicates=True, result_file=None):
        """
        Perform random search with the specified number of trials
        
        Args:
            num_trials: Number of parameter combinations to try
            prevent_duplicates: If True, ensures no parameter set is tried twice
        """
        results = []
        tried_configs = set()  # Keep track of configs we've already tried
        if result_file:
            results = self._load_results(result_file)
            tried_configs = set(_extract_configs(results))

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

            if self.feature_dims is not None:
                params['mlp_feature_dims'] = {}
                num_selected_features = np.random.randint(1, len(self.feature_dims) + 1)
                selected_features = np.random.choice(list(self.feature_dims.keys()), num_selected_features, replace=False)

                for feature in selected_features:
                    input_dim = self.feature_dims[feature][0]
                    output_dims = self.feature_dims[feature][1]
                    idx = np.random.randint(0, len(output_dims))
                    params['mlp_feature_dims'][feature] = (input_dim, output_dims[idx])

            params_hashable = _dict_to_hashable(params)
            
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

    def _load_results(self, filepath):
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
        results = self._load_results(results_file)
        
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
        results = self._load_results(results_file)
                
        # Convert results to DataFrame for easier analysis
        rows = []
        for experiment in results:
            params = experiment['params']
            metrics = experiment['metrics']['k=10']  # Use k=10 metrics
            
            row = {**params, **metrics}
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
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