import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Read existing dataset
print("Reading existing dataset...")
df_existing = pd.read_csv('btc_financial_data.csv')
df_existing['Date'] = pd.to_datetime(df_existing['Date'])
df_existing.set_index('Date', inplace=True)

# Get date range from existing data
start_date = df_existing.index.min()
end_date = df_existing.index.max()

print(f"Collecting new external factors from {start_date.date()} to {end_date.date()}")
print("=" * 60)

# New External Factors tickers - using yfinance format
ef_tickers = [
    'YM=F',      # YM (Dow Jones Futures)
    'CNH=X',     # CNHUSD (USD/CNH)
    'NKD=F',     # NKD (Nikkei 225 Futures)
    'NQ=F',      # NQ (Nasdaq 100 Futures)
    'GC=F',      # GC (Gold Futures)
    'FEX=F',     # FEX (if available)
    'JPY=X',     # JPYUSD (USD/JPY)
    'ZC=F',      # Z (Corn Futures)
    'CL=F',      # CL (Crude Oil Futures)
    'DX-Y.NYB'   # DXY (US Dollar Index)
]

ef_names = [
    'YM', 'CNHUSD', 'NKD', 'NQ', 'GC', 
    'FEX', 'JPYUSD', 'Z', 'CL', 'DXY'
]

# Collect External Factors data (Close prices only)
print("\nCollecting External Factors data...")
ef_data = pd.DataFrame()

for ticker, name in zip(ef_tickers, ef_names):
    try:
        print(f"   Downloading {name} ({ticker})...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            ef_data[f'{name}-EF'] = data['Close']
            print(f"   ✓ {name}: {len(data)} rows collected")
        else:
            print(f"   ⚠ Warning: No data available for {ticker}")
    except Exception as e:
        print(f"   ✗ Error downloading {ticker}: {e}")

# Drop old -EF columns from existing dataset
old_ef_cols = [c for c in df_existing.columns if '-EF' in c]
print(f"\nRemoving {len(old_ef_cols)} old external factor columns...")
df_existing = df_existing.drop(columns=old_ef_cols)

# Merge new external factors with existing data
print(f"Merging {len(ef_data.columns)} new external factor columns...")
df_updated = pd.concat([ef_data, df_existing], axis=1)

# Forward fill external factors
print("Applying forward fill to external factors...")
for col in ef_data.columns:
    df_updated[col] = df_updated[col].ffill()

# Reorder columns: EF first, then BPI, then TI
ef_cols = [c for c in df_updated.columns if '-EF' in c]
bpi_cols = [c for c in df_updated.columns if '-BPI' in c]
ti_cols = [c for c in df_updated.columns if '-TI' in c]
df_updated = df_updated[ef_cols + bpi_cols + ti_cols]

# Reset index to make Date a column
df_updated.reset_index(inplace=True)

# Save updated dataset
output_file = 'btc_financial_data.csv'
df_updated.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("✓ External factors updated successfully!")
print(f"✓ Total rows: {len(df_updated)}")
print(f"✓ Total columns: {len(df_updated.columns)}")
print(f"✓ Saved to: {output_file}")

print("\nColumn breakdown:")
print(f"  - External Factors (-EF): {len(ef_cols)} columns")
print(f"  - Bitcoin Price Info (-BPI): {len(bpi_cols)} columns")
print(f"  - Technical Indicators (-TI): {len(ti_cols)} columns")

print("\nExternal Factor columns:")
for col in ef_cols:
    missing = df_updated[col].isnull().sum()
    print(f"  {col:15s}: {missing:5d} missing")

print("\nFirst 5 rows preview:")
print(df_updated.head())
