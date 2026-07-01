"""
Merge blockchain on-chain metrics into the existing btc_financial_data.csv.
- Aggregates sub-daily data to daily (mean)
- Renames columns with -BC suffix
- Joins on date
- Saves merged file
"""

import pandas as pd
import numpy as np

# Paths
BLOCKCHAIN_CSV = "/Users/gracegooneratne/Desktop/Neuralnet Daily data/blockchain_combined.csv"
FINANCIAL_CSV = "/Users/gracegooneratne/Desktop/neuralNet chainData/bitcoin_daily_classifier/data/raw/btc_financial_data.csv"
OUTPUT_CSV = "/Users/gracegooneratne/Desktop/neuralNet chainData/bitcoin_daily_classifier/data/raw/btc_financial_data.csv"

def main():
    # Load blockchain data
    bc = pd.read_csv(BLOCKCHAIN_CSV)
    print(f"Blockchain raw: {bc.shape[0]} rows, {bc.shape[1]} cols")
    print(f"  Columns: {list(bc.columns)}")

    # Aggregate to daily by taking the mean per date
    metric_cols = [c for c in bc.columns if c not in ('unix_timestamp', 'date')]
    for col in metric_cols:
        bc[col] = pd.to_numeric(bc[col], errors='coerce')

    bc_daily = bc.groupby('date')[metric_cols].mean().reset_index()
    print(f"Blockchain daily: {bc_daily.shape[0]} rows")

    # Rename columns with -BC suffix
    rename_map = {}
    for col in metric_cols:
        rename_map[col] = col + "-BC"
    bc_daily = bc_daily.rename(columns=rename_map)
    bc_daily = bc_daily.rename(columns={'date': 'Date'})
    print(f"  BC columns: {[c for c in bc_daily.columns if c != 'Date']}")

    # Load existing financial data
    fin = pd.read_csv(FINANCIAL_CSV)
    print(f"\nFinancial data: {fin.shape[0]} rows, {fin.shape[1]} cols")

    # Check if -BC columns already exist (avoid duplicates on re-run)
    existing_bc = [c for c in fin.columns if c.endswith('-BC')]
    if existing_bc:
        print(f"  Dropping existing BC columns: {existing_bc}")
        fin = fin.drop(columns=existing_bc)

    # Merge on Date
    merged = fin.merge(bc_daily, on='Date', how='left')
    print(f"\nMerged: {merged.shape[0]} rows, {merged.shape[1]} cols")

    # Report coverage
    bc_cols = [c for c in merged.columns if c.endswith('-BC')]
    for col in bc_cols:
        non_null = merged[col].notna().sum()
        print(f"  {col}: {non_null}/{len(merged)} non-null ({non_null/len(merged)*100:.1f}%)")

    # Forward-fill then back-fill remaining NaNs in BC columns
    for col in bc_cols:
        merged[col] = merged[col].ffill().bfill()
    
    print("\nAfter forward-fill:")
    for col in bc_cols:
        non_null = merged[col].notna().sum()
        print(f"  {col}: {non_null}/{len(merged)} non-null")

    # Save
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Total columns: {len(merged.columns)}")
    print(f"New columns: {bc_cols}")


if __name__ == "__main__":
    main()
