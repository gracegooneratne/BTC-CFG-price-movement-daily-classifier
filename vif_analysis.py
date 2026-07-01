"""
VIF (Variance Inflation Factor) analysis for feature selection.
Removes highly multicollinear features to reduce noise.

Usage: Modify the feature list and data file path in main() as needed.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DailyBitcoinDataLoader
from src.data.preprocessor import DailyDataPreprocessor


def compute_vif(X, feature_names):
    """Compute VIF for each feature."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = feature_names
    vif_data['VIF'] = [
        variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])
    ]
    return vif_data.sort_values('VIF', ascending=False)


def iterative_vif_removal(X, feature_names, threshold=10.0):
    """Iteratively remove features with VIF > threshold."""
    remaining = list(feature_names)
    X_current = X.copy()
    removed = []
    
    while True:
        vif_df = compute_vif(X_current, remaining)
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            break
        
        worst = vif_df.iloc[0]
        print(f"  Removing {worst['Feature']} (VIF={worst['VIF']:.1f})")
        removed.append(worst['Feature'])
        
        idx = remaining.index(worst['Feature'])
        remaining.pop(idx)
        X_current = np.delete(X_current, idx, axis=1)
    
    return remaining, removed, X_current


def main():
    # Load Bitcoin data
    loader = DailyBitcoinDataLoader('data/raw')
    df = loader.load_and_prepare_data(filename='btc_financial_data.csv', add_lags=True, lag_days=1)
    
    # Define all features to analyze (modify as needed)
    # This is the full 31-feature set
    feature_names = [
        # External factors (8)
        'YM-EF', 'NKD-EF', 'NQ-EF', 'GC-EF', 'JPYUSD-EF', 'Z-EF', 'CL-EF', 'DXY-EF',
        # Bitcoin OHLCV (5)
        'Open-BPI', 'High-BPI', 'Low-BPI', 'Close-BPI', 'Volume-BPI',
        # Technical indicators (17)
        'AD-TI', 'BBands_Upper-TI', 'BBands_Middle-TI', 'BBands_Lower-TI',
        'EMA_12-TI', 'EMA_26-TI', 'MACD-TI', 'MACD_Signal-TI', 'MACD_Hist-TI',
        'NATR-TI', 'OBV-TI', 'RSI-TI', 'SMA_20-TI', 'SMA_50-TI',
        'Stoch_K-TI', 'Stoch_D-TI', 'Stoch_Signal-TI',
        # Lagged (1)
        'Close_lag1', 'PriceChange_lag1'
    ]
    
    available_features = [f for f in feature_names if f in df.columns]
    
    preprocessor = DailyDataPreprocessor(target_variable='price_direction', scale_features=False)
    X, y = preprocessor.fit_transform(df, available_features, 'price_direction')
    
    print(f"Starting features: {len(available_features)}")
    print(f"\nInitial VIF values:")
    vif_df = compute_vif(X, available_features)
    for _, row in vif_df.iterrows():
        marker = " ***HIGH***" if row['VIF'] > 10 else ""
        print(f"  {row['Feature']:25s} VIF={row['VIF']:10.1f}{marker}")
    
    print(f"\n{'='*60}")
    print(f"Iterative VIF removal (threshold=10):")
    print(f"{'='*60}")
    remaining, removed, X_reduced = iterative_vif_removal(X, available_features, threshold=10.0)
    
    print(f"\nRemoved {len(removed)} features: {removed}")
    print(f"Remaining {len(remaining)} features: {remaining}")
    
    # Save results
    with open('results/step2_vif_features.txt', 'w') as f:
        f.write("REMAINING FEATURES (VIF < 10):\n")
        for feat in remaining:
            f.write(f"  {feat}\n")
        f.write(f"\nREMOVED FEATURES:\n")
        for feat in removed:
            f.write(f"  {feat}\n")
    
    print(f"\nFeature list saved to results/step2_vif_features.txt")


if __name__ == "__main__":
    main()
