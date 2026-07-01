"""
Data preprocessing for daily Bitcoin classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class DailyDataPreprocessor:
    """Preprocess daily Bitcoin data for classification."""
    
    def __init__(self, target_variable: str = 'price_direction', scale_features: bool = True):
        """
        Args:
            target_variable: Name of target column
            scale_features: Whether to standardize features
        """
        self.target_variable = target_variable
        self.scale_features = scale_features
        self.scaler_X = StandardScaler() if scale_features else None
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Prepare feature matrix."""
        df = df.copy()
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"Warning: Missing features: {missing}")
        
        self.feature_names = available_features
        return df[available_features]
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: list, 
                     target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit scaler and transform data."""
        if target_col is None:
            target_col = self.target_variable
        
        X_df = self.prepare_features(df, feature_cols)
        X = X_df.values
        
        if self.scale_features and self.scaler_X is not None:
            X = self.scaler_X.fit_transform(X)
        
        y = None
        if target_col in df.columns:
            y = df[target_col].values
        
        return X, y
    
    def transform(self, df: pd.DataFrame, feature_cols: list, 
                  target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using fitted scaler."""
        if target_col is None:
            target_col = self.target_variable
        
        X_df = self.prepare_features(df, feature_cols)
        X = X_df.values
        
        if self.scale_features and self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        
        y = None
        if target_col in df.columns:
            y = df[target_col].values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.85) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets (time series split).
        
        Args:
            X: Feature matrix
            y: Target vector
            train_ratio: Ratio of data to use for training
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
