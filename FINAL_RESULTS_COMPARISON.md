# Final Results Comparison

## All Models - Performance Summary

| Model | Features | Tune Accuracy | Holdout Accuracy (Mean ± Std) |
|-------|----------|---------------|-------------------------------|
| **True Baseline (Random)** | N/A | N/A | **50.40%** |
| **Original 31 Features** | 31 | 67.57% | **50.99%** (single run) |
| **VIF Filtered (12 Features)** | 12 | N/A | **49.05%** (single run) |
| **31 + Blockchain (42 Features)** | 42 | 72.97% | **49.34% ± 4.01%** (5-fold) |
| **VIF Filtered Full (17 Features)** | 17 | 70.27% | **49.67% ± 2.05%** (5-fold) |

---

## Detailed Hyperparameters and Results

### 1. True Baseline (Random Guessing)

**Features**: None  
**Method**: Random 50/50 prediction

**Hyperparameters**: None

**Performance**:
- Tune Accuracy: N/A
- Holdout Accuracy: **50.40%**

**Interpretation**: Pure random guessing baseline.

---

### 2. Original 31 Features (Baseline BNN)

**Features**: 31
- 8 External Factors (YM, NKD, NQ, GC, JPYUSD, Z, CL, DXY)
- 5 Bitcoin OHLCV (Open, High, Low, Close, Volume)
- 16 Technical Indicators (AD, BBands, EMA, MACD, NATR, OBV, RSI, SMA, Stoch)
- 2 Lagged Features (Close_lag1, PriceChange_lag1)

**Optimal Hyperparameters**:
- Architecture: **[256, 16, 16]**
- Activation: **relu**
- Dropout: **0.50**
- Learning rate: **3.76e-05**
- Batch size: **64**
- Epochs: **40**
- Alpha: **0.5052**
- Beta: **9.98e-02**
- Validation split: **0.25**

**Performance**:
- Tune Accuracy: **67.57%**
- Holdout Accuracy: **50.99%** (single run)
- Overfitting gap: **16.58 pp**

**Interpretation**: Massive overfitting. Performs no better than random.

---

### 3. VIF Filtered (12 Features)

**Features**: 12 (removed 19 features with VIF > 10)
- 4 External Factors (NKD, GC, Z, CL, DXY)
- 1 Bitcoin OHLCV (Volume)
- 5 Technical Indicators (MACD_Signal, NATR, OBV, RSI, Stoch_K)
- 1 Lagged Feature (PriceChange_lag1)

**Removed Features** (VIF > 10):
- All price features (Open, High, Low, Close, Close_lag1)
- All moving averages (SMA_20, SMA_50, EMA_12, EMA_26)
- All Bollinger Bands
- MACD components (MACD, MACD_Hist)
- Correlated external factors (YM, NQ, JPYUSD)
- Other (AD, Stoch_D, NKD)

**Optimal Hyperparameters**: None (used fixed architecture)

**Performance**:
- Tune Accuracy: N/A
- Holdout Accuracy: **49.05%** (single run, full dataset)

**Interpretation**: VIF reduction does NOT improve performance. Still random.

---

### 4. 31 + Blockchain (42 Features)

**Features**: 42 (31 baseline + 11 blockchain)
- All 31 baseline features
- 11 Blockchain metrics:
  - n-transactions, n-transactions-excluding-popular
  - difficulty, hash-rate
  - trade-volume, miners-revenue, market-cap
  - avg-block-size, n-transactions-per-block
  - median-confirmation-time, cost-per-transaction-percent

**Optimal Hyperparameters**:
- Architecture: **[512, 128]**
- Activation: **elu**
- Dropout: **0.15**
- Learning rate: **4.90e-05**
- Batch size: **128**
- Epochs: **85**
- Alpha: **0.7956**
- Beta: **1.90e-06**
- Validation split: N/A (not in search space)

**Performance**:
- Tune Accuracy: **72.97%**
- Holdout Accuracy: **49.34% ± 4.01%** (5-fold)
- Overfitting gap: **23.63 pp**

**Interpretation**: Adding blockchain features INCREASES overfitting but does NOT improve holdout performance.

---

### 5. VIF Filtered Full (17 Features)

**Features**: 17 (VIF filtered from 42 features)
- 4 External Factors (GC, Z, CL, DXY)
- 1 Bitcoin OHLCV (Volume)
- 5 Technical Indicators (MACD_Signal, NATR, OBV, RSI, Stoch_K)
- 6 Blockchain metrics (trade-volume, miners-revenue, avg-block-size, median-confirmation-time, cost-per-transaction-percent, n-transactions)
- 1 Lagged Feature (PriceChange_lag1)

**Removed Features** (VIF > 10, 25 total):
- All price features (Open, High, Low, Close, Close_lag1)
- All moving averages (SMA, EMA)
- All Bollinger Bands
- MACD components
- Correlated external factors (YM, NQ, JPYUSD, NKD)
- 5 blockchain metrics (market-cap, difficulty, hash-rate, n-transactions-excluding-popular, n-transactions-per-block)
- Other (AD, Stoch_D)

**Optimal Hyperparameters**:
- Architecture: **[512, 256, 512]**
- Activation: **elu**
- Dropout: **0.20**
- Learning rate: **9.35e-05**
- Batch size: **64**
- Epochs: **70**
- Alpha: **0.6841**
- Beta: **1.31e-05**
- Validation split: **0.10**

**Performance**:
- Tune Accuracy: **70.27%**
- Holdout Accuracy: **49.67% ± 2.05%** (5-fold)
- Overfitting gap: **20.60 pp**

**Interpretation**: VIF + blockchain still performs at random chance.

---

## Key Findings

### 1. All Models Perform at Random Chance
- True baseline (random): **50.40%**
- All BNN models: **49-51%**
- **No model beats random guessing**

### 2. Massive Overfitting
- Tune accuracies: **67-73%**
- Holdout accuracies: **49-51%**
- Overfitting gap: **17-24 percentage points**

### 3. Adding Blockchain Features Does NOT Help
- 31 features: 50.99%
- 42 features (+ blockchain): 49.34% ± 4.01%
- **Blockchain features add noise, not signal**

### 4. VIF Feature Reduction Does NOT Help
- 31 features: 50.99%
- 12 features (VIF filtered): 49.05%
- 42 features: 49.34% ± 4.01%
- 17 features (VIF filtered): 49.67% ± 2.05%
- **Removing multicollinear features does not improve performance**

### 5. Model Complexity Does NOT Matter
- Simple [256, 16, 16]: 50.99%
- Complex [512, 256, 512]: 49.67%
- **Architecture has no impact when features lack predictive power**

---

## Statistical Significance

### Confidence Intervals (95%)
- **31 + Blockchain**: 41.48% - 57.20% (wide, includes 50%)
- **VIF Full**: 45.64% - 53.70% (narrower, still includes 50%)

### Standard Deviations
- **31 + Blockchain**: ± 4.01 pp (high variance)
- **VIF Full**: ± 2.05 pp (lower variance, but still random)

**Conclusion**: No model is statistically significantly different from 50% (random chance).

---

## Final Conclusion

**The features used in this study lack predictive power for Bitcoin daily price direction.**

1. ✅ **Proper methodology**: 80/20 split, 100 Optuna trials, 5-fold validation
2. ✅ **Comprehensive feature sets**: Tested 12, 17, 31, and 42 features
3. ✅ **Multiple approaches**: Baseline, VIF filtering, blockchain integration
4. ✅ **Rigorous validation**: Multiple independent runs, confidence intervals

**Result**: All approaches fail. Performance = random guessing.

**Implication**: Bitcoin daily price direction is not predictable using:
- Traditional market indicators (equity indices, commodities, FX)
- Technical indicators (MACD, RSI, Bollinger Bands, etc.)
- On-chain blockchain metrics (hash rate, transactions, difficulty)
- Lagged price features

**Recommendation**: Either:
1. Use different features (sentiment, order book, high-frequency data)
2. Use different prediction targets (volatility, longer horizons)
3. Accept that daily Bitcoin price direction is fundamentally unpredictable

---

**Generated**: 2026-06-16
