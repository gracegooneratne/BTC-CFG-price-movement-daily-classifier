## Table of Contents
1. Overview
2. Features and Data Collection
3. Hyperparameter Optimization
4. Results Summary

## Overview
This project is a neural network intended to predict daily price movement in Bitcoin and the altcoin Centrifuge. The model was trained on a shifting 200 day window, where each day’s features are the input vectors and price direction, either up or down, is the target variable. The model is therefore trained with 200 samples, after which it is tested on the data of the 201st day, and then the sample period rolls forward one day, and the model is tested on the 202nd day. The hyperparameters are tuned on a set consisting of the first 80% of the data (with a step size of 50), and are tuned using a Tree-structured Parzen estimator. Then the model is tested on the final 20% of data that was held out. Because the starting weights are random a five-fold test is run and then averaged to get the final accuracy. 




## Features and Data Collection
Two assets were used in this study: Bitcoin (BTC) and Centrifuge (CFG). For Bitcoin, data was collected over the period spanning February 3, 2019 to February 3, 2026, while Centrifuge data covers July 15, 2021 to March 15, 2026, reflecting the shorter trading history available for the latter asset.
The feature set was organized into three categories:
External market factors — eight variables capturing broader macroeconomic and market conditions.
Coin-specific data — OHLCV (Open, High, Low, Close, Volume) data for each asset, along with 16 technical indicators derived from these price series.
On-chain metrics — 11 blockchain-derived metrics, applicable to Bitcoin only, as comparable on-chain data was not available for Centrifuge.
All coin-specific and external market data were retrieved using the yfinance library. On-chain metrics were sourced separately via the blockchain.info API, and were used exclusively in the Bitcoin prediction pipeline due to the lack of an equivalent data source for Centrifuge.
Data collection was implemented across four scripts: fetch_bitcoin_price.py and fetch_cfg_price.py for coin-specific OHLCV data, fetch_external_factors.py for macroeconomic indicators, and fetch_blockchain_metrics.py for Bitcoin on-chain metrics.

### Features
 
**1. External Market Factors (8 columns) — Suffix: `-EF`**
 
> Global market indicators that may influence coin price.
 
| Column | Description |
|--------|-------------|
| `YM-EF` | Dow Jones Mini Futures | 
| `NKD-EF` | Nikkei 225 Futures | 
| `NQ-EF` | NASDAQ-100 Futures | 
| `GC-EF` | Gold Futures | 
| `JPYUSD-EF` | Japanese Yen to USD Exchange Rate |
| `Z-EF` | Soybean Futures (Agricultural commodity proxy) | 
| `CL-EF` | Crude Oil Futures | 
| `DXY-EF` | US Dollar Index | 
 
---
**2. Coin-specific Factors**

a. OHLCV (5 columns) — Suffix: `-BPI/-CD`
 
>Coin Price Index data from exchanges.
 
| Column | Description |
|--------|-------------|
| `Open-BPI/CD` | Opening price for the day | 
| `High-BPI/CD` | Highest price during the day |
| `Low-BPI/CD` | Lowest price during the day | 
| `Close-BPI/CD` | Closing price for the day | 
| `Volume-BPI/CD` | Trading volume |
 
b. Technical Indicators (16 columns) — Suffix: `-TI`**
 
> Computed from price/volume data.
 
| Column | Description |
|--------|-------------|
| `EMA_12-TI` | 12-period Exponential Moving Average |
| `EMA_26-TI` | 26-period Exponential Moving Average |
| `SMA_20-TI` | 20-period Simple Moving Average |
| `SMA_50-TI` | 50-period Simple Moving Average |
| `RSI-TI` | Relative Strength Index (0–100 scale) |
| `MACD-TI` | Moving Average Convergence Divergence |
| `MACD_Signal-TI` | MACD Signal Line (9-period EMA of MACD) |
| `MACD_Hist-TI` | MACD Histogram (MACD − Signal) |
| `Stoch_K-TI` | Stochastic Oscillator %K (fast line) |
| `Stoch_D-TI` | Stochastic Oscillator %D (slow line) |
| `BBands_Upper-TI` | Bollinger Bands Upper Band |
| `BBands_Middle-TI` | Bollinger Bands Middle Band (SMA) |
| `BBands_Lower-TI` | Bollinger Bands Lower Band |
| `NATR-TI` | Normalized Average True Range (volatility) |
| `OBV-TI` | On-Balance Volume (cumulative volume flow) |
| `AD-TI` | Accumulation/Distribution Line |
 
---
 
**3. (Blockchain Only) On-Chain Metrics (11 columns) — Suffix: `-BC`**
 

| Column | Description | 
|--------|-------------|
| `n-transactions-BC` | Total number of transactions | 
| `n-transactions-excluding-popular-BC` | Transactions excluding popular addresses | 
| `n-transactions-per-block-BC` | Average transactions per block | 
| `trade-volume-BC` | Total BTC traded on-chain |
| `difficulty-BC` | Mining difficulty |
| `hash-rate-BC` | Network hash rate |
| `miners-revenue-BC` | Miner revenue (block rewards + fees) |
| `cost-per-transaction-percent-BC` | Transaction cost as % of value | 
| `avg-block-size-BC` | Average block size | 
| `median-confirmation-time-BC` | Median time to confirm transaction | 
| `market-cap-BC` | Bitcoin market capitalization |
 
---
 
## Model and Hyperparameters
The model used was a Neural Network Classifier with Xavier uniform weight initialization for binary classification (Up/Down price direction).


**Architecture Hyperparameters**
Number of layers: 1-3 hidden layers\
Layer sizes: 64, 128, 256, or 512 neurons per layer\
Activation function: ReLU, Tanh, or ELU\


**Regularization Hyperparameters**\
Dropout rate: 0.1 to 0.6 (step: 0.05)\
Alpha (KL divergence weight): 0.1 to 1.0\
Beta (prior precision): 1×10⁻⁶ to 1×10⁻³ (log scale)\


**Optimization Hyperparameters**\
Learning rate: 1×10⁻⁵ to 5×10⁻⁴ (log scale)\
Batch size: 32, 64, or 128\
Epochs: 40 to 120 (step: 10)\
Early stopping patience: 15 to 30 epochs (step: 5)\
Validation split: 0.1 to 0.3 (step: 0.05)\


**Hyperparameter Optimization**\
Method: Optuna with Tree-structured Parzen Estimator (TPE) sampler\
Trials: 100\
Evaluation: Walk-forward validation with 200-day rolling window

 

## Process and Results 
Initially, the model was trained on BTC data. A baseline accuracy of 49.20% was established by measuring the accuracy of a majority-class classifier on the held-out 20% test set.

The hyperparameter tuning → 5-fold testing pipeline was then run on the 31-feature Bitcoin set (external market factors and coin-specific data). This failed to beat the baseline and exhibited a severe overfitting gap, with tuning accuracy substantially exceeding test accuracy. To address this, a VIF analysis was conducted and features with the highest multicollinearity were iteratively removed until the remaining 12 features each had a VIF below 10. Rerunning the pipeline on this reduced feature set yielded no meaningful improvement in test performance, and the overfitting gap persisted.


The feature set was then expanded by incorporating on-chain metrics, bringing the total to 43 features. The pipeline was rerun but again failed to exceed random guessing and continued to exhibit severe overfitting. A second VIF analysis reduced the set to 17 features; however, neither test accuracy nor the overfitting gap improved.


Attention then shifted to the altcoin Centrifuge, on the basis that its less efficient market might offer greater opportunity for a significant result. A majority-class baseline of 58.61% was established using the same methodology. Running the pipeline on the full 31-feature CFG set again produced results below 50% with severe overfitting. Following VIF filtering down to 13 features, test accuracy improved to 56.2% and the overfitting gap narrowed — however, this still fell short of the majority-class baseline. A confusion matrix revealed that the model was predicting Down approximately 84% of the time regardless of the true label, confirming that it had not learned a meaningful signal but was instead exploiting the class imbalance.


Across all configurations tested, the model consistently failed to demonstrate meaningful predictive power over daily Bitcoin and Centrifuge price direction, suggesting that cryptocurrency price movement may not be reliably learnable from the feature sets and architectures explored here, a finding that is consistent with the broader literature on market efficiency.





 




| Model Features | Architecture | Regularization Parameters | Training Parameters | Tune Accuracy | Test Accuracy |
|---|---|---|---|---|---|
|BTC Baseline (majority-class)| | | | | 49.20% |
|BTC 31 feature set|[256, 16, 16] with ReLu |0.5,0.505,0.0998 |3.762 $\times$ 10-5, 64, 40, 15, .25  |67.57% |48.34% |
|BTC VIF-filter on 31 feature set (31→12)|[128, 64, 128] with ELU |0.4, 0.931, 1.728 * 10-5 |4.328 * 10-4, 128, 40, 15, .1| 72.97%|50.46%|
|BTC 31 feature set + on-chain features|[512, 128] with ELU|0.15, 0.796, 1.904 * 10-6|4.902 * 10-5, 128, 85, 25, 0.1 |72.97% |49.34% |
|BTC VIF-filter on 31 + on chain feature set (42→ 17)| [512, 256, 512] with ELU|0.2, 0.684, 1.31 *10-5|9.349*10-5, 64, 70, 25, 0.1 |70.27% |49.67% |
|CFG Baseline (majority-class)| | | | | 58.61% |
|CFG 31 feature set (no on-chain features) |[128,512] with ELU |0.55, 0.265, 2.742 *10-5 |1.185*10-5, 64, 110, 25, .3| 65.2%| 48.4%| 
|CFG VIF-filtered on 31 feature set (31→13) |[256, 64, 64] with ReLu | 0.3, 0.835, 4.284*10-6|1.013*10-4, 123, 70, 20, .26|65.2%|56.20% | 


<details>
<summary>Limitations</summary>

Hyperparameter tuning used step_size=50 (2% coverage) for computational efficiency. While this enabled rapid exploration of the hyperparameter space, it may have introduced variance in hyperparameter selection. A large portion of the data is consequently underutilized. The consistent poor performance on the holdout set (equivalent or lower than baseline standards across all configurations) suggests this limitation did not materially affect the conclusion that the model lacks predictive power. However, in retrospect, experimentations with step size should have been explored. 

</details>


