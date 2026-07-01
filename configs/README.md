# Experiment Configurations

This directory contains YAML configuration files for all experiments.

## Usage

Run any experiment using the generalizable training script:

```bash
python train_experiment.py --config configs/<experiment>.yaml
```

## Available Experiments

### Bitcoin Experiments

1. **btc_31_features.yaml** - Bitcoin 31 Features (Baseline)
   - 8 External Factors
   - 5 Bitcoin OHLCV
   - 16 Technical Indicators
   - 1 Lagged feature
   - **Total: 30 features** (31 with target)

2. **btc_42_features.yaml** - Bitcoin 42 Features (with On-Chain)
   - 31 baseline features
   - 11 On-chain blockchain metrics
   - **Total: 42 features**

3. **btc_vif_31_to_12.yaml** - Bitcoin VIF-Filtered (31→12)
   - VIF threshold: 10.0
   - Removed 19 features with high multicollinearity
   - **Total: 12 features**

4. **btc_vif_42_to_17.yaml** - Bitcoin VIF-Filtered (42→17)
   - VIF threshold: 10.0
   - Removed 25 features with high multicollinearity
   - **Total: 17 features**

### Centrifuge (CFG) Experiments

5. **cfg_31_features.yaml** - Centrifuge 31 Features (Baseline)
   - 8 External Factors
   - 5 CFG OHLCV
   - 16 Technical Indicators
   - 1 Lagged feature
   - **Total: 30 features**

6. **cfg_vif_31_to_13.yaml** - Centrifuge VIF-Filtered (31→13)
   - VIF threshold: 10.0
   - Removed 18 features with high multicollinearity
   - **Total: 13 features**

## Configuration Format

Each YAML file contains:

```yaml
experiment_name: "Descriptive Name"
asset: "Asset Name (Ticker)"

data:
  file: "path/to/data.csv"
  target_column: "Close-BPI" or "Close-CD"
  
features:
  - "Feature1"
  - "Feature2"
  # ... list all features

output:
  results_file: "results/experiment_name.json"
  model_prefix: "experiment_prefix"
```

## Results

All experiment results are saved to the `results/` directory with the filename specified in each config.

## Example Commands

```bash
# Run Bitcoin 31 features baseline
python train_experiment.py --config configs/btc_31_features.yaml

# Run CFG VIF-filtered experiment
python train_experiment.py --config configs/cfg_vif_31_to_13.yaml

# Run Bitcoin with on-chain features
python train_experiment.py --config configs/btc_42_features.yaml
```

## Methodology

All experiments follow the same methodology:
1. Load data and features from config
2. Split 80/20 train/holdout
3. Optuna hyperparameter tuning (100 trials) on 80% tune set
4. 5-fold validation on 20% holdout set
5. Save results to JSON file

This ensures consistent, reproducible experiments across all feature sets and assets.
