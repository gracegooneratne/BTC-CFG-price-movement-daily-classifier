"""
Classification evaluation metrics for daily Bitcoin price direction prediction.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def compute_roc_auc(y_true, y_proba):
    """Compute ROC AUC from probability predictions."""
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return 0.5  # Default if only one class present

def compute_confusion_mat(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Compute all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)
        
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        'Accuracy': compute_accuracy(y_true, y_pred),
        'Precision': compute_precision(y_true, y_pred),
        'Recall': compute_recall(y_true, y_pred),
        'F1': compute_f1(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['ROC_AUC'] = compute_roc_auc(y_true, y_proba)
    
    return metrics


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
