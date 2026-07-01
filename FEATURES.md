# Feature Documentation

Complete list of all features used in the Bitcoin Daily Price Direction Classifier.

**Dataset**: `btc_financial_data.csv`  
**Date Range**: 2019-03-02 to 2026-03-02 (7 years)  
**Total Features**: 42 (before feature engineering)  
**After Lagging**: 43 features (31 used in baseline model)

---

## Feature Groups

### 1. External Market Factors (8 features)
**Suffix**: `-EF`  
**Source**: Yahoo Finance API (`yfinance`)  
**Purpose**: Global financial indicators that may influence Bitcoin price

| Feature | Ticker | Description | Asset Class |
|---------|--------|-------------|-------------|
| `YM-EF` | `YM=F` | **Dow Jones E-mini Futures** | US Equity Index |
| `NKD-EF` | `NKD=F` | **Nikkei 225 Futures** | Japanese Equity Index |
| `NQ-EF` | `NQ=F` | **NASDAQ-100 E-mini Futures** | US Tech Equity Index |
| `GC-EF` | `GC=F` | **Gold Futures** | Precious Metal / Safe Haven |
| `JPYUSD-EF` | `JPY=X` | **Japanese Yen / USD Exchange Rate** | Currency Pair |
| `Z-EF` | `ZC=F` | **Corn Futures** | Agricultural Commodity |
| `CL-EF` | `CL=F` | **Crude Oil Futures (WTI)** | Energy Commodity |
| `DXY-EF` | `DX-Y.NYB` | **US Dollar Index** | Currency Strength |

**Rationale**: 
- Equity indices capture risk appetite
- Gold and DXY capture safe-haven flows
- Oil and corn represent commodity markets
- JPY/USD represents FX market sentiment

---

### 2. Bitcoin OHLCV (5 features)
**Suffix**: `-BPI`  
**Source**: Yahoo Finance API (`BTC-USD`)  
**Purpose**: Raw Bitcoin price and volume data

| Feature | Description | Unit |
|---------|-------------|------|
| `Open-BPI` | Opening price for the day | USD |
| `High-BPI` | Highest price during the day | USD |
| `Low-BPI` | Lowest price during the day | USD |
| `Close-BPI` | Closing price for the day | USD |
| `Volume-BPI` | Total trading volume | USD |

**Note**: BPI = Bitcoin Price Index

---

### 3. Technical Indicators (16 features)
**Suffix**: `-TI`  
**Source**: Calculated from Bitcoin OHLCV using `ta-lib` / `pandas-ta`  
**Purpose**: Momentum, trend, and volatility indicators

#### Volume Indicators
| Feature | Full Name | Description |
|---------|-----------|-------------|
| `AD-TI` | **Accumulation/Distribution** | Cumulative volume-weighted price momentum |
| `OBV-TI` | **On-Balance Volume** | Cumulative volume flow (up days add, down days subtract) |

#### Volatility Indicators
| Feature | Full Name | Description |
|---------|-----------|-------------|
| `BBands_Upper-TI` | **Bollinger Bands (Upper)** | SMA(20) + 2 * StdDev |
| `BBands_Middle-TI` | **Bollinger Bands (Middle)** | 20-day Simple Moving Average |
| `BBands_Lower-TI` | **Bollinger Bands (Lower)** | SMA(20) - 2 * StdDev |
| `NATR-TI` | **Normalized Average True Range** | ATR normalized by price (volatility %) |

#### Trend Indicators
| Feature | Full Name | Description |
|---------|-----------|-------------|
| `EMA_12-TI` | **Exponential Moving Average (12-day)** | Short-term trend |
| `EMA_26-TI` | **Exponential Moving Average (26-day)** | Medium-term trend |
| `SMA_20-TI` | **Simple Moving Average (20-day)** | Medium-term trend |
| `SMA_50-TI` | **Simple Moving Average (50-day)** | Long-term trend |

#### Momentum Indicators
| Feature | Full Name | Description |
|---------|-----------|-------------|
| `MACD-TI` | **MACD Line** | EMA(12) - EMA(26) |
| `MACD_Signal-TI` | **MACD Signal Line** | EMA(9) of MACD |
| `MACD_Hist-TI` | **MACD Histogram** | MACD - Signal (convergence/divergence) |
| `RSI-TI` | **Relative Strength Index** | Momentum oscillator (0-100) |
| `Stoch_K-TI` | **Stochastic Oscillator %K** | Fast stochastic (0-100) |
| `Stoch_D-TI` | **Stochastic Oscillator %D** | Slow stochastic (SMA of %K) |

**Calculation Details**:
- All moving averages use closing price
- RSI period: 14 days
- Stochastic period: 14 days
- Bollinger Bands: 20-day SMA ± 2 standard deviations

---

### 4. Blockchain On-Chain Metrics (11 features)
**Suffix**: `-BC`  
**Source**: Blockchain.info Charts API  
**Purpose**: Bitcoin network health and activity metrics

#### Transaction Metrics
| Feature | Description | Unit |
|---------|-------------|------|
| `n-transactions-BC` | Total number of confirmed transactions per day | Count |
| `n-transactions-excluding-popular-BC` | Transactions excluding top 100 addresses | Count |
| `n-transactions-per-block-BC` | Average transactions per block | Count |
| `median-confirmation-time-BC` | Median time to confirm a transaction | Minutes |
| `cost-per-transaction-percent-BC` | Transaction fees as % of transaction value | Percentage |

#### Network Security Metrics
| Feature | Description | Unit |
|---------|-------------|------|
| `difficulty-BC` | Mining difficulty (how hard to find a block) | Difficulty Units |
| `hash-rate-BC` | Total network hash rate | TH/s (Terahashes/sec) |

#### Economic Metrics
| Feature | Description | Unit |
|---------|-------------|------|
| `trade-volume-BC` | Total BTC traded on-chain | BTC |
| `miners-revenue-BC` | Total miner revenue (block rewards + fees) | USD |
| `market-cap-BC` | Bitcoin market capitalization | USD |

#### Block Metrics
| Feature | Description | Unit |
|---------|-------------|------|
| `avg-block-size-BC` | Average block size | MB |

**Note**: These features were **disabled in the baseline model** due to poor performance, but were tested in VIF analysis.

---

### 5. Lagged Features (2 features)
**Source**: Engineered from existing features  
**Purpose**: Capture temporal dependencies

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `Close_lag1` | Previous day's closing price | `Close-BPI` shifted by 1 day |
| `PriceChange_lag1` | Previous day's price change | `Close-BPI[t-1] - Close-BPI[t-2]` |

**Note**: Only `PriceChange_lag1` is used in the baseline model (31 features).

---

## Target Variable

| Variable | Description | Values |
|----------|-------------|--------|
| `price_direction` | Next day's price direction | 1 = Up (Close[t+1] > Close[t])<br>0 = Down (Close[t+1] ≤ Close[t]) |

**Calculation**:
```python
df['price_tomorrow'] = df['Close-BPI'].shift(-1)
df['price_change'] = df['price_tomorrow'] - df['Close-BPI']
df['price_direction'] = (df['price_change'] > 0).astype(int)
```

**Class Distribution** (80% train set):
- Up (1): 1,021 samples (50.9%)
- Down (0): 985 samples (49.1%)

**Nearly balanced** - no class imbalance issue.

---

## Feature Configurations

### Baseline Model (31 features)
Used in main experiments:
- 8 External Factors (`-EF`)
- 5 Bitcoin OHLCV (`-BPI`)
- 16 Technical Indicators (`-TI`)
- 1 Lagged Feature (`PriceChange_lag1`)
- 0 Blockchain metrics (disabled)

### VIF-Reduced Model (17 features)
After removing multicollinear features (VIF > 10):
- 4 External Factors: `GC-EF`, `Z-EF`, `CL-EF`, `DXY-EF`
- 1 Bitcoin OHLCV: `Volume-BPI`
- 5 Technical Indicators: `MACD_Signal-TI`, `NATR-TI`, `OBV-TI`, `RSI-TI`, `Stoch_K-TI`
- 6 Blockchain metrics: `trade-volume-BC`, `miners-revenue-BC`, `avg-block-size-BC`, `median-confirmation-time-BC`, `cost-per-transaction-percent-BC`, `n-transactions-BC`
- 1 Lagged Feature: `PriceChange_lag1`

**Removed due to high VIF** (25 features):
- Price features (Open, High, Low, Close, Close_lag1) - highly correlated
- Moving averages (SMA_20, SMA_50, EMA_12, EMA_26) - derived from price
- Bollinger Bands (all 3) - derived from SMA_20
- MACD components (MACD, MACD_Hist) - derived from EMAs
- Some external factors (YM-EF, NQ-EF, NKD-EF, JPYUSD-EF) - correlated with each other
- Some blockchain metrics (market-cap-BC, difficulty-BC, hash-rate-BC) - correlated

---

## Data Quality

### Missing Values
- **External Factors**: Some early dates have missing values (forward-filled)
- **Blockchain Metrics**: Complete coverage from Blockchain.info API
- **Bitcoin OHLCV**: Complete coverage from Yahoo Finance

### Date Range
- **Start**: 2019-03-02
- **End**: 2026-03-02
- **Total Days**: 2,509 (after removing last row with no target)
- **Final Samples**: 2,508

### Data Sources Summary
| Source | API/Provider | Reliability | Update Frequency |
|--------|-------------|-------------|------------------|
| Bitcoin OHLCV | Yahoo Finance | High | Daily |
| External Factors | Yahoo Finance | High | Daily |
| Blockchain Metrics | Blockchain.info | High | Daily |
| Technical Indicators | Calculated | N/A | Derived |

---

## Feature Engineering Notes

1. **No normalization in raw data**: Features are stored in original units
2. **Scaling applied during training**: `StandardScaler` applied per rollover window
3. **No feature selection in baseline**: All 31 features used
4. **VIF analysis**: Identified and removed 25 highly correlated features
5. **Temporal ordering preserved**: No shuffling, respects time series nature

---

## References

- **Technical Indicators**: Calculated using standard formulas from technical analysis
- **Blockchain Metrics**: Definitions from Blockchain.info Charts documentation
- **External Factors**: Standard futures/FX tickers from major exchanges

---

**Last Updated**: 2026-06-16
