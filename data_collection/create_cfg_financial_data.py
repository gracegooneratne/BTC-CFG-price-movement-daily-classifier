"""
Create the complete CFG financial dataset by merging:
1. External factors (futures, FX, commodities) from Yahoo Finance
2. CFG OHLCV price data from Yahoo Finance
3. Technical indicators calculated from CFG price data

Output: cfg_financial_data.csv (ready for model training)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
START_DATE = "2021-07-15"
END_DATE = "2026-03-15"
OUTPUT_FILE = "../data/raw/cfg_financial_data.csv"

# External factor tickers
EXTERNAL_TICKERS = {
    'YM=F': 'YM-EF',      # Dow futures
    'NKD=F': 'NKD-EF',    # Nikkei futures
    'NQ=F': 'NQ-EF',      # NASDAQ futures
    'GC=F': 'GC-EF',      # Gold futures
    'JPY=X': 'JPYUSD-EF', # JPY/USD
    'ZN=F': 'Z-EF',       # Treasury futures
    'CL=F': 'CL-EF',      # Oil futures
    'DX-Y.NYB': 'DXY-EF'  # Dollar index
}


def calculate_technical_indicators(df):
    """Calculate technical indicators from OHLCV data."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Accumulation/Distribution (AD)
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    ad = np.cumsum(mfv)
    df['AD-TI'] = ad
    
    # Bollinger Bands (20-day, 2 std)
    sma_20 = pd.Series(close).rolling(window=20).mean()
    std_20 = pd.Series(close).rolling(window=20).std()
    df['BBands_Upper-TI'] = sma_20 + (2 * std_20)
    df['BBands_Middle-TI'] = sma_20
    df['BBands_Lower-TI'] = sma_20 - (2 * std_20)
    
    # Exponential Moving Averages
    df['EMA_12-TI'] = pd.Series(close).ewm(span=12, adjust=False).mean()
    df['EMA_26-TI'] = pd.Series(close).ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD-TI'] = df['EMA_12-TI'] - df['EMA_26-TI']
    df['MACD_Signal-TI'] = df['MACD-TI'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist-TI'] = df['MACD-TI'] - df['MACD_Signal-TI']
    
    # Normalized Average True Range (NATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14).mean()
    df['NATR-TI'] = (atr / close) * 100
    
    # On-Balance Volume (OBV)
    obv = np.zeros(len(close))
    obv[0] = volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    df['OBV-TI'] = obv
    
    # Relative Strength Index (RSI, 14-day)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI-TI'] = 100 - (100 / (1 + rs))
    
    # Simple Moving Averages
    df['SMA_20-TI'] = pd.Series(close).rolling(window=20).mean()
    df['SMA_50-TI'] = pd.Series(close).rolling(window=50).mean()
    
    # Stochastic Oscillator (14-day)
    low_14 = pd.Series(low).rolling(window=14).min()
    high_14 = pd.Series(high).rolling(window=14).max()
    df['Stoch_K-TI'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-10))
    df['Stoch_D-TI'] = df['Stoch_K-TI'].rolling(window=3).mean()
    
    return df


def main():
    print("=" * 70)
    print("  CFG Financial Data Generator")
    print("=" * 70)
    print(f"  Date range: {START_DATE} → {END_DATE}")
    print("=" * 70)
    
    # Step 1: Fetch external factors
    print("\n📊 Step 1: Fetching external factors...")
    external_data = {}
    
    for ticker, col_name in EXTERNAL_TICKERS.items():
        print(f"  Fetching {col_name} ({ticker})...")
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            external_data[col_name] = data['Close']
        except Exception as e:
            print(f"    ⚠️  Error fetching {ticker}: {e}")
            external_data[col_name] = None
    
    # Create external factors DataFrame
    external_df = pd.DataFrame(external_data)
    external_df = external_df.reset_index()
    external_df['Date'] = pd.to_datetime(external_df['Date']).dt.date
    
    print(f"  ✓ Retrieved {len(external_df)} days of external data")
    
    # Step 2: Fetch CFG price data
    print("\n📊 Step 2: Fetching CFG-USD price data...")
    cfg = yf.download('CFG-USD', start=START_DATE, end=END_DATE, progress=False)
    
    if isinstance(cfg.columns, pd.MultiIndex):
        cfg.columns = cfg.columns.get_level_values(0)
    
    cfg = cfg.reset_index()
    cfg['Date'] = pd.to_datetime(cfg['Date']).dt.date
    
    print(f"  ✓ Retrieved {len(cfg)} days of CFG data")
    print(f"    Date range: {cfg['Date'].iloc[0]} to {cfg['Date'].iloc[-1]}")
    
    # Step 3: Calculate technical indicators
    print("\n📊 Step 3: Calculating technical indicators...")
    temp_df = cfg.copy()
    temp_df = calculate_technical_indicators(temp_df)
    
    # Rename CFG OHLCV columns
    cfg_ohlcv = temp_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    cfg_ohlcv = cfg_ohlcv.rename(columns={
        'Open': 'Open-CD',
        'High': 'High-CD',
        'Low': 'Low-CD',
        'Close': 'Close-CD',
        'Volume': 'Volume-CD'
    })
    
    # Extract technical indicators
    ti_cols = [col for col in temp_df.columns if col.endswith('-TI')]
    cfg_ti = temp_df[['Date'] + ti_cols].copy()
    
    print(f"  ✓ Calculated {len(ti_cols)} technical indicators")
    
    # Step 4: Merge all data
    print("\n📊 Step 4: Merging datasets...")
    
    # Merge external factors with CFG OHLCV
    merged = pd.merge(external_df, cfg_ohlcv, on='Date', how='inner')
    
    # Merge with technical indicators
    merged = pd.merge(merged, cfg_ti, on='Date', how='inner')
    
    print(f"  ✓ Merged data: {len(merged)} rows, {len(merged.columns)} columns")
    
    # Step 5: Order columns properly
    print("\n📊 Step 5: Organizing columns...")
    
    # Define column order
    external_cols = list(EXTERNAL_TICKERS.values())
    cfg_cols = ['Open-CD', 'High-CD', 'Low-CD', 'Close-CD', 'Volume-CD']
    
    final_cols = ['Date'] + external_cols + cfg_cols + ti_cols
    merged = merged[final_cols]
    
    # Step 6: Save to CSV
    print("\n📊 Step 6: Saving to CSV...")
    merged.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved complete dataset to {OUTPUT_FILE}")
    print(f"  Total rows: {len(merged)}")
    print(f"  Total columns: {len(merged.columns)}")
    
    print(f"\n📋 Column breakdown:")
    print(f"  - Date: 1 column")
    print(f"  - External Factors (-EF): {len(external_cols)} columns")
    print(f"  - CFG OHLCV (-CD): {len(cfg_cols)} columns")
    print(f"  - Technical Indicators (-TI): {len(ti_cols)} columns")
    
    print(f"\n📅 Date range: {merged['Date'].iloc[0]} to {merged['Date'].iloc[-1]}")
    
    # Check for missing values
    missing = merged.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing values per column:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDataset ready for training:")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Rows: {len(merged)}")
    print(f"  Features: {len(merged.columns) - 1}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
