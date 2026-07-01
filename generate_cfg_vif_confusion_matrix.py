"""
Generate confusion matrix for CFG VIF experiment.
Reruns the 5-fold validation and creates confusion matrix visualization.
"""

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DailyBitcoinDataLoader
from src.data.preprocessor import DailyDataPreprocessor
from src.models.bayesian_classifier import BayesianClassifier
from src.evaluation.rollover import ClassificationRollover
from src.models.trainer import ClassifierTrainer


def plot_confusion_matrix(cm, title, filename):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Normalize to percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
    
    # Plot
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['Down (0)', 'Up (1)'],
                yticklabels=['Down (0)', 'Up (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {filename}")
    plt.close()


def main():
    print("=" * 70)
    print("CFG VIF CONFUSION MATRIX GENERATION")
    print("=" * 70)
    
    # Load config from results file
    with open('results/cfg_vif_31_to_13.json', 'r') as f:
        config = json.load(f)
    
    features = config['vif_analysis']['remaining_features']
    best_params = config['optuna_hyperparameter_tuning']['best_hyperparameters']
    
    print(f"\nFeatures: {len(features)}")
    print(f"Architecture: {best_params['architecture']['hidden_layers']}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("\n📊 Loading data...")
    loader = DailyBitcoinDataLoader('data/raw')
    df = loader.load_and_prepare_data(filename='cfg_financial_data.csv', add_lags=True, lag_days=1)
    
    # Prepare data
    preprocessor = DailyDataPreprocessor(target_variable='price_direction', scale_features=False)
    X, y = preprocessor.fit_transform(df, features, 'price_direction')
    
    print(f"Total samples: {X.shape[0]}")
    
    # 80/20 split (same as original)
    split_idx = int(len(X) * 0.8)
    X_holdout, y_holdout = X[split_idx:], y[split_idx:]
    
    print(f"Holdout set: {X_holdout.shape[0]} samples (20%)")
    
    # Create model factory
    def create_model():
        return BayesianClassifier(
            input_size=len(features),
            hidden_layers=best_params['architecture']['hidden_layers'],
            output_size=2,
            activation=best_params['architecture']['activation'],
            dropout_rate=best_params['regularization']['dropout'],
            prior_scale=1.0
        )
    
    # Training parameters
    train_params = {
        'batch_size': best_params['optimization']['batch_size'],
        'epochs': best_params['optimization']['epochs'],
        'learning_rate': best_params['optimization']['learning_rate'],
        'alpha': best_params['regularization']['alpha'],
        'beta': best_params['regularization']['beta'],
        'early_stopping_patience': best_params['optimization']['patience'],
        'validation_split': best_params['optimization']['val_split'],
    }
    
    # Run 5-fold validation and collect all predictions
    print(f"\n{'='*70}")
    print("RUNNING 5-FOLD VALIDATION")
    print(f"{'='*70}")
    
    rollover = ClassificationRollover(train_window=200, test_window=1, step_size=1)
    
    all_predictions = []
    all_targets = []
    all_accuracies = []
    
    for run in range(1, 6):
        print(f"\nRun {run}/5...")
        results = rollover.predict_rollover(X_holdout, y_holdout, create_model, train_params, device)
        
        all_predictions.append(results['predictions'])
        all_targets.append(results['targets'])
        
        acc = results['overall_metrics']['Accuracy']
        all_accuracies.append(acc)
        print(f"  Accuracy: {acc*100:.2f}%")
    
    # Aggregate all predictions
    all_preds_combined = np.concatenate(all_predictions)
    all_targets_combined = np.concatenate(all_targets)
    
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies, ddof=1)
    
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Mean accuracy:  {mean_acc*100:.2f}%")
    print(f"Std deviation:  {std_acc*100:.2f} pp")
    print(f"Total predictions: {len(all_preds_combined)}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets_combined, all_preds_combined)
    
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX (Aggregated across 5 runs)")
    print(f"{'='*70}")
    print("\n                Predicted")
    print("              Down    Up")
    print(f"Actual Down  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Up    {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*70}")
    print("DETAILED METRICS")
    print(f"{'='*70}")
    print(f"True Negatives (TN):  {tn:5d}  (Correctly predicted Down)")
    print(f"False Positives (FP): {fp:5d}  (Predicted Up, Actually Down)")
    print(f"False Negatives (FN): {fn:5d}  (Predicted Down, Actually Up)")
    print(f"True Positives (TP):  {tp:5d}  (Correctly predicted Up)")
    print()
    print(f"Accuracy:    {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
    print(f"Precision:   {tp/(tp+fp)*100:.2f}% (of predicted Up, how many were correct)")
    print(f"Recall:      {tp/(tp+fn)*100:.2f}% (of actual Up, how many were caught)")
    print(f"Specificity: {tn/(tn+fp)*100:.2f}% (of actual Down, how many were caught)")
    
    # Class distribution
    total_down = tn + fp
    total_up = fn + tp
    pred_down = tn + fn
    pred_up = fp + tp
    
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Actual Down days:    {total_down:5d} ({total_down/(total_down+total_up)*100:.1f}%)")
    print(f"Actual Up days:      {total_up:5d} ({total_up/(total_down+total_up)*100:.1f}%)")
    print(f"Predicted Down days: {pred_down:5d} ({pred_down/(pred_down+pred_up)*100:.1f}%)")
    print(f"Predicted Up days:   {pred_up:5d} ({pred_up/(pred_down+pred_up)*100:.1f}%)")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        'CFG VIF Confusion Matrix (5-Fold Validation)',
        'results/cfg_vif_confusion_matrix.png'
    )
    
    # Save detailed results
    confusion_results = {
        'experiment': 'CFG VIF 31→13 Features',
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'matrix': cm.tolist()
        },
        'metrics_from_cm': {
            'accuracy': float((tp+tn)/(tp+tn+fp+fn)),
            'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0.0,
            'recall': float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0,
            'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0,
            'f1_score': float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) > 0 else 0.0
        },
        'class_distribution': {
            'actual_down': int(total_down),
            'actual_up': int(total_up),
            'predicted_down': int(pred_down),
            'predicted_up': int(pred_up)
        },
        'validation_stats': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'individual_accuracies': [float(a) for a in all_accuracies],
            'total_predictions': int(len(all_preds_combined))
        }
    }
    
    output_file = 'results/cfg_vif_confusion_matrix.json'
    with open(output_file, 'w') as f:
        json.dump(confusion_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to:")
    print(f"  - {output_file}")
    print(f"  - results/cfg_vif_confusion_matrix.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
