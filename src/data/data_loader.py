"""
Data loader for daily Bitcoin price data with external factors and technical indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class DailyBitcoinDataLoader:
    """Load and prepare daily Bitcoin data for classification."""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
    
    def load_daily_data(self, filename: str = "btc_financial_data.csv") -> pd.DataFrame:
        """
        Load daily Bitcoin data with external factors and technical indicators.
        
        Args:
            filename: CSV file name
            
        Returns:
            DataFrame with date index
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, price_col: str = 'Close-BPI') -> pd.DataFrame:
        """
        Create binary target variable for price direction.
        
        Target: 1 if price goes up next day, 0 if down
        
        Args:
            df: DataFrame with price data
            price_col: Column name for closing price
            
        Returns:
            DataFrame with 'price_direction' column
        """
        df = df.copy()
        
        # Calculate next day's price change
        df['price_tomorrow'] = df[price_col].shift(-1)
        df['price_change'] = df['price_tomorrow'] - df[price_col]
        
        # Binary classification: 1 = Up, 0 = Down
        df['price_direction'] = (df['price_change'] > 0).astype(int)
        
        # Drop rows where we don't know tomorrow's price (last row)
        df = df[:-1]
        
        # Drop helper columns
        df = df.drop(columns=['price_tomorrow', 'price_change'])
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, lag_days: int = 1, price_col: str = None) -> pd.DataFrame:
        """
        Add lagged price features for autoregressive prediction.
        
        Args:
            df: DataFrame with price data
            lag_days: Number of days to lag
            price_col: Column name for closing price (auto-detected if None)
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        # Auto-detect price column if not provided
        if price_col is None:
            if 'Close-BPI' in df.columns:
                price_col = 'Close-BPI'
            elif 'Close-CD' in df.columns:
                price_col = 'Close-CD'
            else:
                raise ValueError("Could not find Close-BPI or Close-CD column")
        
        # Lagged closing price
        df[f'Close_lag{lag_days}'] = df[price_col].shift(lag_days)
        
        # Lagged price change
        df[f'PriceChange_lag{lag_days}'] = df[price_col].pct_change(lag_days)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            method: 'drop' or 'ffill'
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if method == 'ffill':
            df = df.ffill()
        elif method == 'drop':
            df = df.dropna()
        
        return df
    
    def load_and_prepare_data(self, 
                             filename: str = "btc_financial_data.csv",
                             add_lags: bool = True,
                             lag_days: int = 1) -> pd.DataFrame:
        """
        Load and prepare complete dataset.
        
        Args:
            filename: CSV file name
            add_lags: Whether to add lagged features
            lag_days: Number of days to lag
            
        Returns:
            Prepared DataFrame
        """
        # Load data
        df = self.load_daily_data(filename)
        
        # Auto-detect price column
        if 'Close-BPI' in df.columns:
            price_col = 'Close-BPI'
        elif 'Close-CD' in df.columns:
            price_col = 'Close-CD'
        else:
            raise ValueError("Could not find Close-BPI or Close-CD column")
        
        # Create target variable
        df = self.create_target_variable(df, price_col=price_col)
        
        # Add lagged features if requested
        if add_lags:
            df = self.add_lagged_features(df, lag_days, price_col=price_col)
        
        # Handle missing values
        df = self.handle_missing_values(df, method='drop')
        
        return df
