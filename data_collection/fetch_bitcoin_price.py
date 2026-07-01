"""
Fetch Bitcoin OHLCV price data from Yahoo Finance.
Creates the Bitcoin Price Index (BPI) columns for the dataset.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
START_DATE = "2019-03-02"
END_DATE = "2026-03-02"
OUTPUT_FILE = "bitcoin_price_data.csv"

def calculate_technical_indicators(df):
    """
    Calculate technical indicators from OHLCV data.
    Returns DataFrame with technical indicator columns.
    """
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
    print("  Bitcoin Price Data Fetcher (Yahoo Finance)")
    print("=" * 70)
    print(f"  Date range: {START_DATE} → {END_DATE}")
    print(f"  Ticker: BTC-USD")
    print("=" * 70)
    
    # Fetch Bitcoin price data
    print("\nFetching Bitcoin OHLCV data from Yahoo Finance...")
    btc = yf.download('BTC-USD', start=START_DATE, end=END_DATE, progress=False)
    
    if btc.empty:
        print("ERROR: No data retrieved for BTC-USD")
        return
    
    print(f"✓ Retrieved {len(btc)} days of data")
    print(f"  Date range: {btc.index[0].date()} to {btc.index[-1].date()}")
    
    # Reset index and prepare DataFrame
    btc = btc.reset_index()
    btc['Date'] = pd.to_datetime(btc['Date']).dt.date
    
    # Rename columns with -BPI suffix
    btc = btc.rename(columns={
        'Open': 'Open-BPI',
        'High': 'High-BPI',
        'Low': 'Low-BPI',
        'Close': 'Close-BPI',
        'Volume': 'Volume-BPI'
    })
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    temp_df = btc.copy()
    temp_df = temp_df.rename(columns={
        'Open-BPI': 'Open',
        'High-BPI': 'High',
        'Low-BPI': 'Low',
        'Close-BPI': 'Close',
        'Volume-BPI': 'Volume'
    })
    
    temp_df = calculate_technical_indicators(temp_df)
    
    # Copy indicator columns back
    for col in temp_df.columns:
        if col.endswith('-TI'):
            btc[col] = temp_df[col]
    
    # Select final columns
    final_cols = ['Date', 'Open-BPI', 'High-BPI', 'Low-BPI', 'Close-BPI', 'Volume-BPI',
                  'AD-TI', 'BBands_Upper-TI', 'BBands_Middle-TI', 'BBands_Lower-TI',
                  'EMA_12-TI', 'EMA_26-TI', 'MACD-TI', 'MACD_Signal-TI', 'MACD_Hist-TI',
                  'NATR-TI', 'OBV-TI', 'RSI-TI', 'SMA_20-TI', 'SMA_50-TI',
                  'Stoch_K-TI', 'Stoch_D-TI']
    
    btc = btc[final_cols]
    
    # Save to CSV
    btc.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved Bitcoin data to {OUTPUT_FILE}")
    print(f"  Total rows: {len(btc)}")
    print(f"  Total columns: {len(btc.columns)}")
    print(f"\nColumn breakdown:")
    print(f"  - Bitcoin OHLCV (-BPI): 5 columns")
    print(f"  - Technical Indicators (-TI): 16 columns")
    
    print(f"\nFirst 5 rows:")
    print(btc.head())
    
    print(f"\nLast 5 rows:")
    print(btc.tail())
    
    # Check for missing values
    missing = btc.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values per column:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")


if __name__ == "__main__":
    main()
