# Script Usage Guide

## 🎯 Main Scripts (Use These)

### **1. Generic Optuna Tuning**
```bash
python optuna_tune_generic.py
```
**Purpose**: Tune hyperparameters for ANY feature set
- Automatically detects features from config + CSV
- Works for 31, 42, VIF-reduced, or custom feature sets
- Runs 100 Optuna trials + 5-fold validation
- Saves results to `results/optuna_tuning.json` and `results/optuna_5fold.json`

**Replaces**: `optuna_tune.py`, `optuna_tune_v2.py`, `optuna_tune_bc.py`

---

### **2. VIF Analysis Scripts**

#### **VIF Baseline (31 → 12 features)**
```bash
python vif_baseline_optuna.py
```
- Loads 31 baseline features
- Removes features with VIF > 10
- Tunes on reduced feature set
- Results: `results/vif_baseline_5fold.json`

#### **VIF Full (42 → 17 features)**
```bash
python vif_full_optuna.py
```
- Loads 42 features (31 + 11 blockchain)
- Removes features with VIF > 10
- Tunes on reduced feature set
- Results: `results/vif_full_5fold.json`

---

### **3. Specific Feature Set Validation**

#### **Original 31 Features - 5-Fold**
```bash
python optuna_baseline_5fold.py
```
- Uses best params from previous tuning
- Runs 5-fold validation on holdout set
- Results: `results/optuna_baseline_5fold.json`

---

### **4. True Baselines**
```bash
python true_baseline.py
```
- Majority class predictor
- Random 50/50 predictor
- Momentum predictor (yesterday's direction)
- Results: `results/true_baseline.json`

---

## 📊 Visualization Scripts

### **Generate Charts**
```bash
python generate_visualizations.py
```
- Creates plots for results comparison
- Generates charts for write-up

### **Visualize Model Weights**
```bash
python visualize_weights.py
```
- Analyzes Bayesian network weights
- Shows uncertainty distributions

---

## 🔄 Typical Workflow

### **New Feature Set**
1. Add features to `config/config.yaml` or CSV
2. Run `python optuna_tune_generic.py`
3. Results automatically saved

### **VIF Analysis**
1. Run `python vif_baseline_optuna.py` (for baseline)
2. Run `python vif_full_optuna.py` (for full set)
3. Compare results

### **Compare to Baseline**
1. Run `python true_baseline.py`
2. Compare BNN results to random/majority

---

## 📁 Output Files

### **Tuning Results**
- `results/optuna_tuning.json` - Best hyperparameters
- `results/optuna_5fold.json` - 5-fold validation results

### **VIF Results**
- `results/vif_baseline_5fold.json` - VIF baseline (12 features)
- `results/vif_full_5fold.json` - VIF full (17 features)

### **Baseline Results**
- `results/true_baseline.json` - Random/majority baselines
- `results/optuna_baseline_5fold.json` - Original 31 features

### **Holdout Logs**
- `results/optuna_holdout_run1-5.txt` - Individual run logs

---

## ⚙️ Configuration

All scripts automatically read from:
- `config/config.yaml` - Feature definitions
- `data/raw/btc_financial_data.csv` - Data file

To change features:
1. Edit `config/config.yaml` to enable/disable feature groups
2. Or modify CSV to add/remove columns
3. Scripts automatically detect available features

---

## 🎯 One Script to Rule Them All

**`optuna_tune_generic.py`** is now your go-to script for hyperparameter tuning on ANY feature set. No need for separate scripts per configuration!

**Before**: 3 duplicate scripts (optuna_tune.py, optuna_tune_v2.py, optuna_tune_bc.py)  
**After**: 1 generic script that works for everything ✨
