# VIF Analysis Summary

## Overview

Two VIF analyses were performed to reduce multicollinearity:

1. **VIF on Baseline Features (12 features)** - Completed
2. **VIF on Full Feature Set (17 features)** - Running

---

## Analysis 1: VIF on Baseline Features Only

**Input**: 31 baseline features (no blockchain)
- 8 External Factors
- 5 Bitcoin OHLCV  
- 16 Technical Indicators
- 1 Lagged Feature (PriceChange_lag1)
- 1 Lagged Feature (Close_lag1)

**VIF Threshold**: 10.0

**Removed**: 19 features with VIF > 10

**Remaining**: 12 features

### Remaining Features (12)
| Category | Features |
|----------|----------|
| **External Factors (4)** | NKD-EF, GC-EF, Z-EF, CL-EF, DXY-EF |
| **Bitcoin OHLCV (1)** | Volume-BPI |
| **Technical Indicators (5)** | MACD_Signal-TI, NATR-TI, OBV-TI, RSI-TI, Stoch_K-TI |
| **Lagged (1)** | PriceChange_lag1 |
| **Blockchain (0)** | None (not included in baseline) |

### Results
- **Holdout Accuracy**: 49.05% (single run, full dataset)
- **Conclusion**: No improvement over baseline

**Files**:
- `vif_analysis.py` - Analysis script
- `results/step2_vif_results.json` - Results
- `results/step2_vif_features.txt` - Feature list
- `results/step2_vif_log.txt` - Detailed log

---

## Analysis 2: VIF on Full Feature Set (42 features)

**Input**: 42 total features (31 baseline + 11 blockchain)
- 8 External Factors
- 5 Bitcoin OHLCV
- 16 Technical Indicators
- 2 Lagged Features
- 11 Blockchain Metrics

**VIF Threshold**: 10.0

**Removed**: 25 features with VIF > 10

**Remaining**: 17 features

### Remaining Features (17)
| Category | Count | Features |
|----------|-------|----------|
| **External Factors** | 4 | GC-EF, Z-EF, CL-EF, DXY-EF |
| **Bitcoin OHLCV** | 1 | Volume-BPI |
| **Technical Indicators** | 5 | MACD_Signal-TI, NATR-TI, OBV-TI, RSI-TI, Stoch_K-TI |
| **Blockchain** | 6 | trade-volume-BC, miners-revenue-BC, avg-block-size-BC, median-confirmation-time-BC, cost-per-transaction-percent-BC, n-transactions-BC |
| **Lagged** | 1 | PriceChange_lag1 |

### Methodology
1. ✅ VIF analysis on all 42 features
2. ✅ Remove features with VIF > 10 iteratively
3. 🔄 80/20 train/holdout split
4. 🔄 Optuna hyperparameter tuning (100 trials, step=50)
5. ⏳ 5-fold validation on holdout set (step=1)

### Results (In Progress)
- **Tune Accuracy**: TBD
- **Holdout Accuracy (5-fold)**: TBD
- **Mean ± Std**: TBD

**Files**:
- `vif_full_optuna.py` - Analysis script
- `results/vif_full_optuna_tuning.json` - Tuning results
- `results/vif_full_5fold.json` - Final 5-fold results
- `results/vif_full_holdout_run1-5.txt` - Individual run logs

---

## Comparison: VIF Baseline vs VIF Full

| Metric | VIF Baseline (12 feat) | VIF Full (17 feat) |
|--------|------------------------|---------------------|
| **Initial Features** | 31 | 42 |
| **Removed** | 19 | 25 |
| **Remaining** | 12 | 17 |
| **Blockchain Features** | 0 | 6 |
| **Methodology** | Single run, full data | 80/20 + 100 trials + 5-fold |
| **Holdout Accuracy** | 49.05% | TBD |

---

## Features Removed (Common to Both)

### Always Removed (High VIF)
1. **Price Features** (perfect collinearity):
   - Open-BPI, High-BPI, Low-BPI, Close-BPI, Close_lag1
   - VIF: 284K, 1.7K, 284, 4.4K, 3.5K

2. **Moving Averages** (derived from price):
   - SMA_20-TI, SMA_50-TI, EMA_12-TI, EMA_26-TI
   - VIF: ∞, 766, ∞, 24.8K

3. **Bollinger Bands** (derived from SMA):
   - BBands_Upper-TI, BBands_Middle-TI, BBands_Lower-TI
   - VIF: ∞, 9.1K, 15.0

4. **MACD Components** (derived from EMAs):
   - MACD-TI, MACD_Hist-TI
   - VIF: 30.7, ∞

5. **Correlated External Factors**:
   - YM-EF (Dow), NQ-EF (NASDAQ), JPYUSD-EF
   - VIF: 36.5, 75.1, 68.9

6. **Other**:
   - AD-TI (26.1), Stoch_D-TI (13.9), NKD-EF (14.7)

### Blockchain Features Removed (VIF Full only)
1. **market-cap-BC** (VIF: 8995) - Highly correlated with price
2. **difficulty-BC** (VIF: 228) - Correlated with hash rate
3. **hash-rate-BC** (VIF: 42) - Correlated with difficulty
4. **n-transactions-excluding-popular-BC** (VIF: 1215)
5. **n-transactions-per-block-BC** (VIF: 22)

---

## Key Findings

### What VIF Removes
- **Redundant price representations**: Only need one price measure
- **Derived indicators**: Moving averages, Bollinger Bands, MACD
- **Correlated market indices**: Keep only independent ones
- **Correlated blockchain metrics**: Keep only independent ones

### What VIF Keeps
- **Independent external factors**: Gold, Oil, Corn, Dollar Index
- **Volume**: Trading volume (not correlated with price)
- **Pure momentum indicators**: RSI, Stochastic %K
- **Volatility**: NATR (normalized)
- **Volume flow**: OBV, MACD Signal
- **Independent blockchain metrics**: 6 metrics with VIF < 10
- **Temporal dependency**: Yesterday's price change

### Impact on Performance
- VIF reduction does NOT improve accuracy
- Both configurations perform at ~49% (random)
- The problem is lack of predictive signal, not multicollinearity

---

## Deleted Files (VIF + Blockchain, 17 features - OLD)

The following files were deleted as they used an inconsistent methodology:
- ❌ `vif_bc_analysis.py` - Old VIF with blockchain (no proper split)
- ❌ `vif_bc_optuna.py` - Old VIF+BC with Optuna
- ❌ `results/vif_bc_*` - All old VIF+BC results

**Reason**: These used VIF on baseline + blockchain but didn't follow the same rigorous 80/20 + 100 trials + 5-fold methodology.

---

## Current Status

✅ **VIF Baseline (12 features)**: Complete
🔄 **VIF Full (17 features)**: Running (100 trials + 5-fold validation)

**Expected completion**: ~3-4 hours

---

**Last Updated**: 2026-06-16
