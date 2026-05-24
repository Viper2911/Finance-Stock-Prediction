# 📈 Advanced AI Market Forecast App

An interactive, machine-learning-powered web dashboard that predicts future stock prices and volatility ranges for 18 major global companies. 

Unlike standard time-series tutorials, this project enhances the Prophet forecasting engine by injecting custom technical indicators (Relative Strength Index and Simple Moving Averages) as external regressors, creating a much smarter, market-aware AI.

---

## ✨ Key Features

* **Multi-Variate AI Forecasting:** Uses a customized Prophet engine, upgraded with `SMA_20` and `RSI_14` technical indicators to forecast 1 to 30 days into the future.
* **Volatility Ranges:** Automatically calculates the 80% confidence interval, providing an Expected Price, a Worst-Case Floor (`yhat_lower`), and a Best-Case Ceiling (`yhat_upper`).
* **Live Currency Conversion:** Automatically fetches the live USD-to-INR exchange rate. All US tech and finance stocks are seamlessly converted and displayed in Indian Rupees (₹) for a unified user experience.
* **Interactive Visuals:** Institutional-grade, interactive Plotly charts featuring historical overlays and draggable range sliders.
* **Global Universe:** Supports 18 hand-picked global titans across US Tech, US Finance, and the Indian NIFTY 50.

---

## 🧠 How It Works Under the Hood (Architecture)

This application doesn't just draw a line of best fit; it processes live market data through a 5-step pipeline every time a user makes a request.

### 1. Live Data Ingestion (yfinance)
When a user selects a stock, the app instantly pings the Yahoo Finance API to download historical daily trading data from 2015 to the present day.

### 2. Currency Normalization Pipeline
If the selected stock is an American company (like Apple or Nvidia), the app makes a secondary API call to fetch the live USD/INR exchange rate. It then multiplies the entire historical dataframe (Open, High, Low, Close) by this rate. This ensures the AI trains on, and the user views, a pure Rupee (₹) dataset.

### 3. Feature Engineering (Technical Indicators)
Before feeding data to the AI, we calculate two institutional-grade metrics using raw Pandas math:
* **20-Day Simple Moving Average (SMA_20):** Identifies the baseline trend over the last month, smoothing out daily noise.
* **14-Day Relative Strength Index (RSI_14):** A momentum oscillator that measures the speed and change of price movements. It tells the AI if a stock is mathematically "overbought" (due for a crash) or "oversold" (due for a bounce).

### 4. Machine Learning (Prophet + Custom Regressors)
Standard time-series models only look at Date and Price (Univariate modeling). This custom engine uses advanced regressors. By feeding the `SMA_20` and `RSI_14` columns into the model alongside the price, the AI learns *why* the price is moving. It trains on this multi-variate dataset in real-time, calculates daily seasonality, and projects a future timeframe based on the user's slider input.

### 5. Probabilistic Forecasting
Instead of giving a single, easily disproven number, the model outputs a probabilistic range. It provides `yhat` (the median expected price), alongside `yhat_lower` and `yhat_upper` (the mathematical floor and ceiling representing an 80% confidence interval of where the price will land).


## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning Engine:** Prophet (Customized with Technical Regressors)
* **Data Ingestion:** yfinance
* **Data Manipulation:** pandas, numpy
* **Data Visualization:** plotly

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Viper2911/Finance-Stock-Prediction.git](https://github.com/Viper2911/Finance-Stock-Prediction.git)
   cd Finance-Stock-Prediction
