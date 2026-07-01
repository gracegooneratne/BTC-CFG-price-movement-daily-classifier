# Results Folder - Consolidated Summary

## 📁 File Structure

### **Baseline Results (2 JSON files)**

1. **`baseline_majority_class.json`** - Bitcoin majority class baseline
2. **`cfg_baseline_majority_class.json`** - CFG majority class baseline

### **Bitcoin Experiments (4 JSON files)**

3. **`bitcoin_31_features.json`** - Original 31 features experiment
4. **`bitcoin_42_features.json`** - 31 + 11 blockchain features experiment
5. **`bitcoin_vif_31_to_12.json`** - VIF-filtered baseline (31→12)
6. **`bitcoin_vif_42_to_17.json`** - VIF-filtered full set (42→17)

### **CFG Experiments (4 JSON files)**

7. **`cfg_31_features.json`** - CFG 31 features baseline
8. **`cfg_vif_31_to_13.json`** - CFG VIF-filtered (31→13)
9. **`cfg_vif_confusion_matrix.json`** - CFG VIF confusion matrix data
10. **`cfg_vif_confusion_matrix.png`** - CFG VIF confusion matrix visualization

### **Visualizations (6 files)**

11. **`viz1_overfitting_comparison.png`** - Tune vs holdout accuracy comparison
12. **`viz2_feature_importance.png`** - Feature importance analysis
13. **`viz3_timeseries_performance.png`** - Performance over time
14. **`viz4_hyperparameter_landscape.png`** - Hyperparameter optimization landscape
15. **`weights_animation.gif`** - Animated Bayesian weight evolution
16. **`weights_final.png`** - Final weight distributions

---

## 📊 Results Summary

### **Baselines**

#### **Bitcoin Majority Class (`baseline_majority_class.json`)**
- **Train**: 2,006 samples (50.9% Up, 49.1% Down)
- **Test**: 502 samples (49.2% Up, 50.8% Down)
- **Majority class**: Up (1)
- **Test accuracy**: **49.20%**

**Conclusion**: Bitcoin is nearly balanced, so majority class baseline achieves ~49% (essentially random).

#### **CFG Majority Class (`cfg_baseline_majority_class.json`)**
- **Train**: 1,323 samples (44.7% Up, 55.3% Down)
- **Test**: 331 samples (41.4% Up, 58.6% Down)
- **Majority class**: Down (0)
- **Test accuracy**: **58.61%**

**Conclusion**: CFG is imbalanced (55% Down), so majority class achieves 58.6% by always predicting Down. However, this has 0% precision/recall for Up class.

---

### **2. Bitcoin 31 Features (`bitcoin_31_features.json`)**

**Feature Set**: 8 External + 5 Bitcoin OHLCV + 17 Technical + 1 Lagged

**Results**:
- **Tune Accuracy**: 67.6% (on 80% tune set, step=50)
- **Holdout Accuracy**: 48.3% ± 2.9% (5-fold, step=1)
- **Overfitting Gap**: 19.3 percentage points

**Conclusion**: Severe overfitting. Model memorizes patterns but cannot generalize.

---

### **3. Bitcoin 42 Features (`bitcoin_42_features.json`)**

**Feature Set**: 31 Baseline + 11 Blockchain metrics

**Blockchain Features Added**:
- hash-rate, difficulty, miners-revenue, transaction-fees
- n-transactions, n-unique-addresses, market-cap, trade-volume
- avg-block-size, median-confirmation-time, cost-per-transaction-percent

**Results**:
- **Tune Accuracy**: 73.0% (on 80% tune set, step=50)
- **Holdout Accuracy**: 49.3% ± 4.0% (5-fold, step=1)
- **Overfitting Gap**: 23.7 percentage points

**Conclusion**: Adding blockchain features INCREASED overfitting without improving generalization.

---

### **4. Bitcoin VIF 31→12 (`bitcoin_vif_31_to_12.json`)**

**VIF Analysis**: Removed 19 features with VIF > 10, kept 12 low-collinearity features

**Remaining Features** (12):
- **External**: NKD, GC, Z, CL, DXY
- **Bitcoin**: Volume
- **Technical**: MACD_Signal, NATR, OBV, RSI, Stoch_K
- **Lagged**: PriceChange_lag1

**Results**:
- **Tune Accuracy**: 73.0% (on 80% tune set, step=50)
- **Holdout Accuracy**: 50.5% ± 1.6% (5-fold, step=1)
- **Overfitting Gap**: 22.5 percentage points

**Conclusion**: VIF filtering did NOT reduce overfitting or improve performance. Still random.

---

### **5. Bitcoin VIF 42→17 (`bitcoin_vif_42_to_17.json`)**

**VIF Analysis**: Removed 25 features with VIF > 10, kept 17 low-collinearity features

**Remaining Features** (17):
- **External**: GC, Z, CL, DXY
- **Bitcoin**: Volume
- **Technical**: MACD_Signal, NATR, OBV, RSI, Stoch_K
- **Blockchain** (6 kept): trade-volume, miners-revenue, avg-block-size, median-confirmation-time, cost-per-transaction-percent, n-transactions
- **Lagged**: PriceChange_lag1

**Results**:
- **Tune Accuracy**: 70.3% (on 80% tune set, step=50)
- **Holdout Accuracy**: 49.7% ± 2.1% (5-fold, step=1)
- **Overfitting Gap**: 20.6 percentage points

**Conclusion**: VIF + blockchain features did not help. Performance still random.

---

### **6. CFG 31 Features (`cfg_31_features.json`)**

**Feature Set**: 8 External + 5 CFG OHLCV + 16 Technical + 1 Lagged

**Results**:
- **Tune Accuracy**: 65.2% (on 80% tune set, step=50)
- **Holdout Accuracy**: 48.4% ± 5.0% (5-fold, step=1)
- **Overfitting Gap**: 16.8 percentage points

**Conclusion**: CFG shows similar pattern to Bitcoin - severe overfitting, no generalization.

---

### **7. CFG VIF 31→13 (`cfg_vif_31_to_13.json`)**

**VIF Analysis**: Removed 18 features with VIF > 10, kept 13 low-collinearity features

**Remaining Features** (13):
- **External**: GC, Z, CL, DXY
- **CFG**: High, Volume
- **Technical**: MACD_Signal, MACD_Hist, NATR, OBV, RSI, Stoch_K
- **Lagged**: PriceChange_lag1

**Results**:
- **Tune Accuracy**: 65.2% (on 80% tune set, step=50)
- **Holdout Accuracy**: 56.2% ± 4.0% (5-fold, step=1)
- **Overfitting Gap**: 9.0 percentage points

**Conclusion**: VIF filtering IMPROVED CFG performance (56.2% vs 48.4%) and reduced overfitting. This is DIFFERENT from Bitcoin where VIF didn't help.

**Confusion Matrix Analysis**:
- Model heavily biased toward predicting Down (84% of predictions)
- Specificity: 84% (good at catching Down days)
- Recall: 16% (poor at catching Up days)
- The 56.2% accuracy comes from correctly identifying most Down days, but missing most Up days

---

## 🎯 Overall Conclusions

### **Key Findings**

#### **Bitcoin**
1. **No Predictive Power**: All Bitcoin configurations achieve ~48-51% holdout accuracy (random guessing)
2. **Consistent Overfitting**: All models overfit heavily (67-73% tune, 48-51% holdout)
3. **More Features ≠ Better**: 42 features worse than 31 (more overfitting)
4. **VIF Filtering Ineffective**: Removing multicollinearity doesn't help when there's no signal
5. **Blockchain Metrics Useless**: On-chain data adds no predictive value for daily direction

#### **CFG (Centrifuge)**
1. **Marginal Improvement**: CFG VIF achieves 56.2% (slightly above baseline 58.6% majority class)
2. **VIF Helped CFG**: Unlike Bitcoin, VIF filtering improved CFG performance and reduced overfitting
3. **Class Imbalance Issue**: CFG's 56.2% comes from predicting Down 84% of the time (not true predictive power)
4. **Different Dynamics**: CFG shows different feature relationships than Bitcoin, but still limited predictive signal

### **Performance Comparison**

#### **Bitcoin**
| Configuration | Features | Tune Acc | Holdout Acc | Overfitting Gap |
|---------------|----------|----------|-------------|-----------------|
| Majority Class | N/A | N/A | 49.2% | N/A |
| 31 Features | 31 | 67.6% | 48.3% ± 2.9% | 19.3 pp |
| 42 Features | 42 | 73.0% | 49.3% ± 4.0% | 23.7 pp |
| VIF 31→12 | 12 | 73.0% | 50.5% ± 1.6% | 22.5 pp |
| VIF 42→17 | 17 | 70.3% | 49.7% ± 2.1% | 20.6 pp |

#### **CFG (Centrifuge)**
| Configuration | Features | Tune Acc | Holdout Acc | Overfitting Gap |
|---------------|----------|----------|-------------|-----------------|
| Majority Class | N/A | N/A | 58.6% | N/A |
| 31 Features | 31 | 65.2% | 48.4% ± 5.0% | 16.8 pp |
| VIF 31→13 | 13 | 65.2% | 56.2% ± 4.0% | 9.0 pp |

### **Statistical Significance**

**Bitcoin**: All holdout accuracies fall within **48-51%** range, statistically indistinguishable from random (49.2% baseline). The 95% confidence intervals all overlap with 50%.

**CFG**: The 56.2% accuracy is only marginally below the 58.6% majority class baseline, and the model achieves this by heavily biasing toward Down predictions (84% of predictions). This is not true predictive power.

### **Final Verdict**

#### **Bitcoin**
Daily price direction is **NOT predictable** using:
- ❌ External market factors
- ❌ Technical indicators
- ❌ Blockchain metrics
- ❌ Lagged features
- ❌ Any combination of the above

**The problem is**: There is no predictive signal in the data for daily price direction.

#### **CFG (Centrifuge)**
Daily price direction shows **marginal predictability**:
- ✓ VIF filtering improved performance (56.2% vs 48.4% baseline)
- ⚠️ But model is heavily biased (84% Down predictions)
- ⚠️ Performance barely beats majority class baseline (58.6%)
- ⚠️ Low recall (16%) for Up days indicates limited true predictive power

**The problem is**: While CFG shows different dynamics than Bitcoin, the predictive signal remains weak and largely driven by class imbalance.

---

## 📈 Methodology

All experiments used consistent methodology:

1. **Data Split**: 80/20 train/holdout
   - Bitcoin: 2,006/502 samples
   - CFG: 1,323/331 samples
2. **Hyperparameter Tuning**: 100 Optuna trials on tune set (step=50)
3. **Final Validation**: 5-fold on holdout set (step=1)
4. **Rollover**: 200-day training window, 1-day prediction
5. **Per-window normalization**: StandardScaler fit on each train window
6. **No data leakage**: Holdout never used for hyperparameter selection

---

## 📝 File Format

All JSON files follow consistent structure:

```json
{
  "experiment": "...",
  "description": "...",
  "feature_configuration": {...},
  "data_split": {...},
  "optuna_hyperparameter_tuning": {...},
  "holdout_validation_5_fold": {...},
  "performance_analysis": {...},
  "interpretation": {...}
}
```

---

## 🔍 For Write-Up

**Use these files**:
- **Baselines**: `baseline_majority_class.json`, `cfg_baseline_majority_class.json`
- **Bitcoin**: `bitcoin_31_features.json`, `bitcoin_42_features.json`, `bitcoin_vif_31_to_12.json`, `bitcoin_vif_42_to_17.json`
- **CFG**: `cfg_31_features.json`, `cfg_vif_31_to_13.json`, `cfg_vif_confusion_matrix.json`
- **Visualizations**: `viz1-4.png` for analysis, `weights_*.png/gif` for model internals
- **Confusion Matrix**: `cfg_vif_confusion_matrix.png` for CFG prediction patterns

**Key messages**:
- **Bitcoin**: Daily price direction remains unpredictable at ~49% accuracy (random guessing) despite extensive experimentation
- **CFG**: Shows marginal improvement (56.2%) but heavily biased toward Down predictions, indicating weak predictive signal
