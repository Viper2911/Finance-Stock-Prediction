# 📈 Advanced AI Market Forecast App

An interactive, deep-learning-powered web dashboard that predicts future stock prices and volatility ranges for major Indian companies. 

This project utilizes a custom **Long Short-Term Memory (LSTM)** neural network trained with a multi-task quantile loss function. Instead of standard time-series regression, it evaluates a 10-day sliding window of engineered technical features to provide probabilistic market forecasting.

---

## ✨ Key Features

* **Deep Learning Forecasting:** Powered by a custom-trained multi-head LSTM model built with TensorFlow/Keras.
* **Multi-Task Quantile Regression:** The network outputs a directional bias alongside an 80% confidence volatility interval, providing an Expected Price, a Worst-Case Floor (`ret_low`), and a Best-Case Ceiling (`ret_high`).
* **Feature Engineering Pipeline:** Instantly processes raw market data to calculate moving averages (SMA_20), momentum (RSI_14), and structural proximity to all-time highs/lows.
* **Interactive Visuals:** Institutional-grade, interactive Plotly charts featuring historical overlays and draggable range sliders.
* **NIFTY-Focused Universe:** Supports 19 hand-picked titans across the Indian stock market, including Reliance, TCS, Sun Pharma, Bajaj Finance, and Jio Financial Services.

---

## 🧠 How It Works Under the Hood (Architecture)

### 1. Live Data Ingestion (yfinance)
When a user selects a stock, the app pings the Yahoo Finance API to download up to 10 years of historical daily trading data to establish long-term structural anchors.

### 2. Feature Engineering
Before feeding data to the neural network, raw prices are converted into five scaled features:
* **20-Day SMA & 14-Day RSI:** Captures recent momentum and mean-reversion levels.
* **ATH & ATL Proximity:** Normalizes how close the current price is to historical boundaries.
* **Intraday Trend:** Evaluates the close-to-open delta.

### 3. Matrix Formatting
The engine groups the scaled features into a 3D matrix representing a 10-day rolling window `(1, 10, 5)`. This ensures the LSTM can "remember" the exact shape of the trend over the last two trading weeks rather than just looking at a single day in a vacuum.

### 4. Multi-Head Predictions
The pre-trained network (`quantile_market_model.h5`) processes the sequence and outputs four distinct predictions simultaneously: 
1. The probability of an upward movement tomorrow.
2. The 10th percentile expected return (Floor).
3. The 50th percentile expected return (Median).
4. The 90th percentile expected return (Ceiling).

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning Engine:** TensorFlow, Keras (LSTM)
* **Data Pipelines:** yfinance, pandas, numpy, scikit-learn
* **Data Visualization:** plotly

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Viper2911/Finance-Stock-Prediction.git](https://github.com/Viper2911/Finance-Stock-Prediction.git)
   cd Finance-Stock-Prediction
