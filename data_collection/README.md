# Data Collection Scripts

This folder contains all scripts used to fetch and prepare the raw data for the Bitcoin Neural Network classifier.

## Overview

The data collection pipeline consists of three main sources that are merged together:

1. **Bitcoin Price Data** (OHLCV + Technical Indicators)
2. **External Market Factors** (Global financial indicators)
3. **Blockchain On-Chain Metrics** (Network statistics)

---

## Scripts

### 1. `fetch_bitcoin_price.py`

**Purpose**: Fetch Bitcoin OHLCV data and calculate technical indicators

**Source**: Yahoo Finance API (`BTC-USD`)

**Output**: `bitcoin_price_data.csv` (22 columns)
- 5 OHLCV columns (Open-BPI, High-BPI, Low-BPI, Close-BPI, Volume-BPI)
- 16 Technical Indicators (AD-TI, BBands, EMA, MACD, RSI, etc.)

**Usage**:
```bash
python fetch_bitcoin_price.py
```

**Date Range**: 2019-03-02 to 2026-03-02 (7 years)

---

### 2. `fetch_external_factors.py`

**Purpose**: Fetch global market indicators that may influence Bitcoin price

**Source**: Yahoo Finance API (`yfinance`)

**Output**: Updates `btc_financial_data.csv` with 8 external factor columns

**Tickers Fetched**:
| Ticker | Column | Description |
|--------|--------|-------------|
| `YM=F` | YM-EF | Dow Jones Mini Futures |
| `NKD=F` | NKD-EF | Nikkei 225 Futures |
| `NQ=F` | NQ-EF | NASDAQ-100 Futures |
| `GC=F` | GC-EF | Gold Futures |
| `JPY=X` | JPYUSD-EF | Japanese Yen / USD Exchange Rate |
| `ZC=F` | Z-EF | Corn Futures (agricultural commodity proxy) |
| `CL=F` | CL-EF | Crude Oil Futures (WTI) |
| `DX-Y.NYB` | DXY-EF | US Dollar Index |

**Usage**:
```bash
python fetch_external_factors.py
```

**Note**: This script reads an existing `btc_financial_data.csv` and replaces the external factor columns with fresh data.

---

### 3. `fetch_blockchain_metrics.py`

**Purpose**: Fetch Bitcoin blockchain on-chain metrics

**Source**: Blockchain.info Charts API (`https://api.blockchain.info/charts/`)

**Output**: `blockchain_combined.csv` (13 columns)

**Metrics Fetched** (11 total):
| Metric | Description |
|--------|-------------|
| `n-transactions` | Total number of transactions per day |
| `n-transactions-excluding-popular` | Transactions excluding popular addresses |
| `difficulty` | Mining difficulty |
| `trade-volume` | Total BTC traded on-chain |
| `miners-revenue` | Miner revenue (block rewards + fees) |
| `market-cap` | Bitcoin market capitalization |
| `avg-block-size` | Average block size in MB |
| `n-transactions-per-block` | Average transactions per block |
| `median-confirmation-time` | Median time to confirm transaction |
| `hash-rate` | Network hash rate (TH/s) |
| `cost-per-transaction-percent` | Transaction cost as % of value |

**Usage**:
```bash
python fetch_blockchain_metrics.py
```

**Rate Limiting**: 3-second delay between API requests

---

### 4. `merge_blockchain_data.py`

**Purpose**: Merge blockchain metrics into the main dataset

**Process**:
1. Reads `blockchain_combined.csv`
2. Aggregates to daily (mean per date)
3. Renames columns with `-BC` suffix
4. Merges with `btc_financial_data.csv` on Date
5. Forward-fills and back-fills missing values
6. Saves to project folder: `../bitcoin_daily_classifier/data/raw/btc_financial_data.csv`

**Usage**:
```bash
python merge_blockchain_data.py
```

**Note**: This script overwrites the final dataset file in the project folder.

---

## Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Fetch Bitcoin Price + Technical Indicators         │
│ Script: fetch_bitcoin_price.py                             │
│ Output: bitcoin_price_data.csv (22 columns)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Fetch External Market Factors                      │
│ Script: fetch_external_factors.py                          │
│ Output: btc_financial_data.csv (30 columns)             │
│         = 8 EF + 5 BPI + 16 TI + Date                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Fetch Blockchain On-Chain Metrics                  │
│ Script: fetch_blockchain_metrics.py                        │
│ Output: blockchain_combined.csv (13 columns)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Merge All Data Sources                             │
│ Script: merge_blockchain_data.py                           │
│ Output: ../bitcoin_daily_classifier/data/raw/              │
│         btc_financial_data.csv (41 columns)             │
│         = 8 EF + 5 BPI + 16 TI + 11 BC + Date              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Model Data Loading (in project)                    │
│ Script: ../bitcoin_daily_classifier/src/data/data_loader.py│
│ Process: Add 2 lagged features (Close_lag1, PriceChange_lag1)│
│ Final: 31 features used in model (29 from CSV + 2 lagged) │
└─────────────────────────────────────────────────────────────┘
```

---

## Running the Complete Pipeline

To refresh all data from scratch:

```bash
# 1. Fetch Bitcoin price and technical indicators
python fetch_bitcoin_price.py

# 2. Fetch external market factors
python fetch_external_factors.py

# 3. Fetch blockchain on-chain metrics
python fetch_blockchain_metrics.py

# 4. Merge blockchain data into final dataset
python merge_blockchain_data.py
```

---

## Data Sources Summary

| Data Type | Source | API/Method | Cost |
|-----------|--------|------------|------|
| Bitcoin OHLCV | Yahoo Finance | `yfinance` library | Free |
| External Factors | Yahoo Finance | `yfinance` library | Free |
| On-Chain Metrics | Blockchain.info | REST API | Free |
| Technical Indicators | Calculated | pandas/numpy | N/A |

---

## Notes

- **No Bitcoin Node Required**: All data is fetched from public APIs
- **Date Range**: 2019-03-02 to 2026-03-02 (7 years of daily data)
- **Blockchain Features**: The 11 blockchain on-chain metrics are included in the CSV but **disabled in the model config** because they reduced performance
- **Weekend/Holiday Handling**: External factors are forward-filled for non-trading days
- **Technical Indicators**: Calculated using standard formulas (RSI, MACD, Bollinger Bands, etc.)

---

## Dependencies

```bash
pip install yfinance pandas numpy requests
```

---

## File Locations

- **Data collection scripts**: `/neuralNet chainData/data_collection/`
- **Intermediate data**: `/Neuralnet Daily data/` (legacy folder, can be archived)
- **Final dataset**: `/neuralNet chainData/bitcoin_daily_classifier/data/raw/btc_financial_data.csv`
