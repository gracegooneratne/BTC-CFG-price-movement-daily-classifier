"""
Rollover (walk-forward) framework for daily Bitcoin price direction classification.
Uses 200-day rolling window with per-window normalization.
"""

import numpy as np
from typing import Dict, Callable, List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.models.bayesian_classifier import BayesianClassifier
from src.models.trainer import ClassifierTrainer
from src.evaluation.metrics import compute_all_metrics


class ClassificationRollover:
    """Rolling window evaluation for classification."""
    
    def __init__(self, train_window: int = 200, test_window: int = 30, step_size: int = 30):
        """
        Args:
            train_window: Number of days for training (200-day window)
            test_window: Number of days to predict
            step_size: Number of days to roll forward
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def generate_rollover_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits for rollover.
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        start = 0
        
        while start + self.train_window + self.test_window <= n_samples:
            train_idx = np.arange(start, start + self.train_window)
            test_idx = np.arange(start + self.train_window, 
                                min(start + self.train_window + self.test_window, n_samples))
            splits.append((train_idx, test_idx))
            start += self.step_size
        
        return splits
    
    def predict_rollover(self, X, y, model_factory, train_params, device='cpu', verbose=True, classification_threshold=0.5) -> Dict:
        """
        Perform rollover classification with per-window normalization.
        
        Args:
            X: Full feature matrix (UNNORMALIZED)
            y: Full target vector (0/1 labels)
            model_factory: Function that creates a new model instance
            train_params: Training parameters
            device: Device to train on
            verbose: Show progress
            
        Returns:
            Dictionary with predictions and metrics
        """
        splits = self.generate_rollover_splits(len(X))
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_metrics = []
        
        iterator = tqdm(splits) if verbose else splits
        
        for train_idx, test_idx in iterator:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Per-window normalization (fit on train, transform both)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Create and train model
            model = model_factory()
            trainer = ClassifierTrainer(
                model, device=device,
                learning_rate=train_params.get('learning_rate', 0.001)
            )
            
            # Separate trainer params from fit params
            fit_params = {k: v for k, v in train_params.items() 
                         if k not in ['learning_rate']}
            fit_params['verbose'] = False
            trainer.fit(X_train, y_train, **fit_params)
            
            # Predict
            y_proba = trainer.predict_proba(X_test)[:, 1]  # Probability of Up
            y_pred = (y_proba >= classification_threshold).astype(int)
            
            # Store results
            all_predictions.append(y_pred)
            all_probabilities.append(y_proba)
            all_targets.append(y_test)
            
            # Compute metrics for this window
            metrics = compute_all_metrics(y_test, y_pred, y_proba)
            all_metrics.append(metrics)
        
        # Aggregate results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Overall metrics
        overall_metrics = compute_all_metrics(all_targets, all_predictions, all_probabilities)
        
        # Average metrics across windows
        avg_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'overall_metrics': overall_metrics,
            'avg_metrics': avg_metrics,
            'window_metrics': all_metrics,
            'n_windows': len(splits)
        }
    
    def print_rollover_summary(self, results: Dict):
        """Print summary of rollover results."""
        print("\n" + "="*60)
        print("Rollover Framework Results (Classification)")
        print("="*60)
        print(f"Number of windows: {results['n_windows']}")
        print(f"Train window size: {self.train_window} days")
        print(f"Test window size: {self.test_window} days")
        print(f"Step size: {self.step_size} days")
        
        # Class distribution
        targets = results['targets']
        preds = results['predictions']
        print(f"\nTarget Distribution:")
        print(f"  Up days (actual):    {(targets == 1).sum()} ({(targets == 1).mean()*100:.1f}%)")
        print(f"  Down days (actual):  {(targets == 0).sum()} ({(targets == 0).mean()*100:.1f}%)")
        print(f"  Up days (predicted): {(preds == 1).sum()} ({(preds == 1).mean()*100:.1f}%)")
        print(f"  Down days (predicted): {(preds == 0).sum()} ({(preds == 0).mean()*100:.1f}%)")
        
        print(f"\nOverall Metrics:")
        print("-"*40)
        for key, value in results['overall_metrics'].items():
            print(f"{key:15s}: {value:10.4f}")
        
        print(f"\nAverage Window Metrics:")
        print("-"*40)
        for key, value in results['avg_metrics'].items():
            print(f"{key:20s}: {value:10.4f}")
