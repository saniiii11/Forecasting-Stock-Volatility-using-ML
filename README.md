## Forecasting Stock Volatility with Deep Learning

**Project originally completed in December 2025.**

This project frames market volatility as a time-series forecasting task, comparing traditional statistical baselines against Deep Learning sequence models to anticipate market uncertainty and identify regime shifts. 

### Data Source

* The project utilizes the [Apple (AAPL) Historical Stock Data](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data?utm_source=chatgpt.com) sourced from Kaggle.
* The dataset contains 2,518 records of historical market data spanning from March 2010 to February 2020.
* Included features are Date, Close/Last, Volume, Open, High, and Low prices.

### Methodology & Feature Engineering

* Computed daily log returns to estimate volatility because they are time-additive and statistically stable.
* Estimated the target variable using a 21-day rolling standard deviation.
* Annualized the volatility by scaling with $\sqrt{252}$ to match standard finance metrics.
* Converted 2D DataFrames into 3D tensors utilizing a sliding window of 21 days with 4 features to train the model.

### Model Architecture

To overcome the vanishing gradient problem in standard RNNs, this project implements a two-layer stacked LSTM network designed to selectively remember important trends and capture non-linear volatility spikes.

* **Input Layer:** 64 LSTM units with `return_sequences=True` and a 20% Dropout rate.
* **Hidden Layer:** 32 LSTM units with a 20% Dropout rate.
* **Output Layer:** Dense layer with 1 neuron.
* **Compilation:** Optimized using Adam and evaluated using Mean Squared Error (MSE).
* **Training:** 100 epochs with a batch size of 32, utilizing Early Stopping with a patience of 10 epochs.

### Model Comparison & Results

The stacked LSTM was evaluated against a standard statistical baseline, GARCH(1,1). While GARCH produces a smooth, mean-reverting curve suitable for stable markets, the LSTM demonstrated superior adaptability in capturing extreme volatility regimes and abrupt market movements.

| Metric | GARCH(1,1) Baseline | Stacked LSTM |
| :--- | :--- | :--- |
| **RMSE** | 0.00354 | 0.00131 |
| **MAE** | 0.00294 | 0.00090 |
| **R-Squared** | 0.6421 | 0.9511 |

