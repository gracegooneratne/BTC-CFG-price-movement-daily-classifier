"""
Majority Class Baseline for CFG.

Determines majority class from 80% training data, evaluates on 20% test data.
"""

import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.data_loader import DailyBitcoinDataLoader
from src.data.preprocessor import DailyDataPreprocessor


def main():
    print("=" * 70)
    print("CFG MAJORITY CLASS BASELINE")
    print("=" * 70)
    
    # Load CFG data
    loader = DailyBitcoinDataLoader('data/raw')
    df = loader.load_and_prepare_data(filename='cfg_financial_data.csv', add_lags=True, lag_days=1)
    
    # Use a simple feature for preprocessing (we only need the target)
    features = ['Close-CD']
    preprocessor = DailyDataPreprocessor(target_variable='price_direction', scale_features=False)
    X, y = preprocessor.fit_transform(df, features, 'price_direction')
    
    # 80/20 split
    split_idx = int(len(X) * 0.8)
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Train set: {len(y_train)} samples (80%)")
    print(f"Test set:  {len(y_test)} samples (20%)")
    
    # Class distribution in training set
    train_up = (y_train == 1).sum()
    train_down = (y_train == 0).sum()
    
    print(f"\nTraining Set Class Distribution:")
    print(f"  Up (1):   {train_up:4d} ({train_up/len(y_train)*100:.1f}%)")
    print(f"  Down (0): {train_down:4d} ({train_down/len(y_train)*100:.1f}%)")
    
    # Determine majority class from training data
    majority_class = 1 if train_up > train_down else 0
    majority_label = 'Up' if majority_class == 1 else 'Down'
    
    print(f"\nMajority class: {majority_class} ({majority_label})")
    
    # Test set distribution
    test_up = (y_test == 1).sum()
    test_down = (y_test == 0).sum()
    
    print(f"\nTest Set Class Distribution:")
    print(f"  Up (1):   {test_up:4d} ({test_up/len(y_test)*100:.1f}%)")
    print(f"  Down (0): {test_down:4d} ({test_down/len(y_test)*100:.1f}%)")
    
    # Predict majority class for all test samples
    print(f"\n{'='*70}")
    print("MAJORITY CLASS BASELINE")
    print(f"{'='*70}")
    print(f"Strategy: Always predict {majority_class} ({majority_label})")
    
    y_pred = np.full_like(y_test, majority_class)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    # Save results
    results = {
        'experiment': 'CFG Majority Class Baseline',
        'asset': 'Centrifuge (CFG-USD)',
        'description': 'Majority class determined from 80% train, evaluated on 20% test',
        'data_split': {
            'train_size': int(len(y_train)),
            'test_size': int(len(y_test)),
            'split_ratio': 0.8
        },
        'class_distribution': {
            'train': {
                'up': int(train_up),
                'down': int(train_down),
                'up_percentage': float(train_up/len(y_train)*100)
            },
            'test': {
                'up': int(test_up),
                'down': int(test_down),
                'up_percentage': float(test_up/len(y_test)*100)
            }
        },
        'majority_class': {
            'class': int(majority_class),
            'label': majority_label
        },
        'results': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    }
    
    output_file = 'results/cfg_baseline_majority_class.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
