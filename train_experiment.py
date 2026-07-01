"""
Generalizable Optuna Hyperparameter Tuning + Model Training Script.

Usage:
    python train_experiment.py --config configs/btc_31_features.yaml
    python train_experiment.py --config configs/cfg_vif_31_to_13.yaml
"""

import argparse
import yaml
import numpy as np
import json
import torch
import optuna
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DailyBitcoinDataLoader
from src.data.preprocessor import DailyDataPreprocessor
from src.models.bayesian_classifier import BayesianClassifier
from src.evaluation.rollover import ClassificationRollover
from src.models.trainer import ClassifierTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def quick_rollover_eval(X, y, model_factory, train_params, train_window=200, step_size=50, device='cpu'):
    """Quick rollover for hyperparameter tuning."""
    n = len(X)
    all_preds, all_targets = [], []
    start = 0
    while start + train_window + 1 <= n:
        train_idx = np.arange(start, start + train_window)
        test_idx = np.array([start + train_window])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = model_factory()
        trainer = ClassifierTrainer(model, device=device,
                                    learning_rate=train_params.get('learning_rate', 0.001))
        fit_params = {k: v for k, v in train_params.items() if k != 'learning_rate'}
        fit_params['verbose'] = False
        trainer.fit(X_train, y_train, **fit_params)
        pred = trainer.predict(X_test)
        all_preds.append(pred)
        all_targets.append(y_test)
        start += step_size
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return (all_preds == all_targets).mean()


def run_experiment(config_path: str):
    """Run complete experiment: Optuna tuning + 5-fold validation."""
    
    # Load configuration
    config = load_experiment_config(config_path)
    exp_name = config['experiment_name']
    asset = config['asset']
    data_file = config['data']['file']
    target_col = config['data']['target_column']
    features = config['features']
    results_file = config['output']['results_file']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"Asset: {asset}")
    print(f"Data: {data_file}")
    print(f"Features: {len(features)}")
    print(f"Results: {results_file}")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data...")
    loader = DailyBitcoinDataLoader(Path(data_file).parent)
    df = loader.load_and_prepare_data(filename=Path(data_file).name, add_lags=True, lag_days=1)
    
    # Filter to available features
    available = [f for f in features if f in df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        print(f"⚠️  Missing features: {missing}")
    
    print(f"Using {len(available)} features")
    
    # Prepare data
    preprocessor = DailyDataPreprocessor(target_variable='price_direction', scale_features=False)
    X, y = preprocessor.fit_transform(df, available, 'price_direction')
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    
    # 80/20 split
    split_idx = int(len(X) * 0.8)
    X_tune, y_tune = X[:split_idx], y[:split_idx]
    X_holdout, y_holdout = X[split_idx:], y[split_idx:]
    
    print(f"\nTune set:    {X_tune.shape[0]} samples (80%)")
    print(f"Holdout set: {X_holdout.shape[0]} samples (20%)")
    
    # Hyperparameter tuning
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING (Optuna - 100 trials)")
    print(f"{'='*70}")
    
    n_feat = X.shape[1]
    
    def objective(trial):
        # Architecture
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            size = trial.suggest_categorical(f'layer_{i}_size', [64, 128, 256, 512])
            hidden_layers.append(size)
        
        # Regularization
        dropout = trial.suggest_float('dropout', 0.1, 0.6, step=0.05)
        beta = trial.suggest_float('beta', 1e-6, 1e-3, log=True)
        alpha = trial.suggest_float('alpha', 0.1, 1.0)
        
        # Optimization
        lr = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epochs = trial.suggest_int('epochs', 40, 120, step=10)
        patience = trial.suggest_int('patience', 15, 30, step=5)
        val_split = trial.suggest_float('val_split', 0.1, 0.3, step=0.05)
        
        # Activation
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
        
        def create_model(hl=hidden_layers, act=activation, dr=dropout):
            return BayesianClassifier(
                input_size=n_feat, hidden_layers=hl, output_size=2,
                activation=act, dropout_rate=dr, prior_scale=1.0
            )
        
        train_params = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'alpha': alpha,
            'beta': beta,
            'early_stopping_patience': patience,
            'validation_split': val_split,
        }
        
        acc = quick_rollover_eval(X_tune, y_tune, create_model, train_params, step_size=50, device=device)
        return acc
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=True,
                   callbacks=[lambda study, trial: print(
                       f"  Trial {trial.number:3d}: acc={trial.value:.4f} (best={study.best_value:.4f})"
                   )])
    
    best = study.best_trial
    print(f"\n{'='*70}")
    print("BEST TRIAL")
    print(f"{'='*70}")
    print(f"Tune accuracy: {best.value:.4f}")
    print(f"Parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    
    # Save tuning results
    n_layers = best.params['n_layers']
    hidden_layers = [best.params[f'layer_{i}_size'] for i in range(n_layers)]
    
    tuning_results = {
        'experiment': exp_name,
        'asset': asset,
        'n_features': n_feat,
        'features': available,
        'best_trial_number': best.number,
        'best_tune_accuracy': float(best.value),
        'best_params': best.params,
        'hidden_layers': hidden_layers,
        'n_trials': len(study.trials)
    }
    
    # 5-fold holdout validation
    print(f"\n{'='*70}")
    print("HOLDOUT VALIDATION (5 runs)")
    print(f"{'='*70}")
    
    def create_best_model():
        return BayesianClassifier(
            input_size=n_feat,
            hidden_layers=hidden_layers,
            output_size=2,
            activation=best.params['activation'],
            dropout_rate=best.params['dropout'],
            prior_scale=1.0
        )
    
    best_train_params = {
        'batch_size': best.params['batch_size'],
        'epochs': best.params['epochs'],
        'learning_rate': best.params['learning_rate'],
        'alpha': best.params['alpha'],
        'beta': best.params['beta'],
        'early_stopping_patience': best.params['patience'],
        'validation_split': best.params['val_split'],
    }
    
    rollover = ClassificationRollover(train_window=200, test_window=1, step_size=1)
    
    holdout_accuracies = []
    for run in range(1, 6):
        print(f"\nRun {run}/5...")
        results = rollover.predict_rollover(X_holdout, y_holdout, create_best_model, best_train_params, device)
        acc = results['overall_metrics']['Accuracy']
        holdout_accuracies.append(acc)
        print(f"  Holdout accuracy: {acc*100:.2f}%")
    
    # Statistics
    mean_acc = np.mean(holdout_accuracies)
    std_acc = np.std(holdout_accuracies, ddof=1)
    
    print(f"\n{'='*70}")
    print("5-FOLD HOLDOUT RESULTS")
    print(f"{'='*70}")
    print(f"Mean accuracy:  {mean_acc*100:.2f}%")
    print(f"Std deviation:  {std_acc*100:.2f} pp")
    print(f"95% CI:         [{(mean_acc-1.96*std_acc)*100:.2f}%, {(mean_acc+1.96*std_acc)*100:.2f}%]")
    
    # Save final results
    final_results = {
        'experiment': exp_name,
        'asset': asset,
        'n_features': n_feat,
        'features': available,
        'best_params': best.params,
        'hidden_layers': hidden_layers,
        'tune_accuracy': float(best.value),
        'holdout_runs': {
            f'run_{i}': float(acc) for i, acc in enumerate(holdout_accuracies, 1)
        },
        'statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'min_accuracy': float(np.min(holdout_accuracies)),
            'max_accuracy': float(np.max(holdout_accuracies)),
            'confidence_interval_95': {
                'lower': float(mean_acc - 1.96*std_acc),
                'upper': float(mean_acc + 1.96*std_acc)
            }
        },
        'interpretation': {
            'overfitting_gap': float(best.value - mean_acc),
            'performance_vs_random': float(mean_acc - 0.5)
        }
    }
    
    # Save results
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {results_file}")
    print(f"{'='*70}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Run experiment with config file')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML file')
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == "__main__":
    main()
