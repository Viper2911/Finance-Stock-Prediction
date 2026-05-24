import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objs as go

st.set_page_config(page_title="Market Forecast AI", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📈 Deep Learning Market Forecast</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #888888; margin-bottom: 2rem;'>Powered by Custom LSTM Neural Network Architecture</h5>", unsafe_allow_html=True)

stocks = (
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
    'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'SUNPHARMA.NS', 
    'LAURUSLABS.NS', 'DRREDDY.NS', 'NESTLEIND.NS', 'HINDUNILVR.NS', 
    'BEL.NS', 'IOC.NS', 'BAJFINANCE.NS', 'JIOFIN.NS', 'CDSL.NS'
)

@st.cache_resource
def load_ml_assets():
    model = load_model("quantile_market_model.h5", compile=False)
    scaler = joblib.load("market_scaler.pkl")
    return model, scaler

@st.cache_data
def load_and_engineer_data(ticker):
    df = yf.download(ticker, period="10y", progress=False)
    
    if df.empty:
        st.error("No data found.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Historical_ATH"] = df["High"].cummax()
    df["Historical_ATL"] = df["Low"].cummin()
    df["ATH_Proximity"] = df["Close"] / df["Historical_ATH"]
    df["ATL_Proximity"] = df["Close"] / df["Historical_ATL"]
    df["Intraday_Trend"] = (df["Close"] - df["Open"]) / df["Open"]
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    change = df["Close"].diff()
    gain = change.mask(change < 0, 0.0)
    loss = -change.mask(change > 0, 0.0)
    rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_stock = st.selectbox("🔍 Search & Select Market Asset", stocks)

try:
    model, scaler = load_ml_assets()
    st.toast("✅ Neural Network & Scaler Online", icon="🧠")
except Exception as e:
    st.error(f"⚠️ Failed to load model/scaler.\n\nError: {e}")
    st.stop()

with st.spinner("Fetching live market data and engineering features..."):
    df = load_and_engineer_data(selected_stock)

with st.container(border=True):
    st.subheader(f"Historical Trend: {selected_stock.replace('.NS', '')}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close Price", line=dict(color="#00ffcc", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="20-Day SMA", line=dict(color="#ff9900", dash="dot", width=2)))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=True,
        height=600,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.subheader("🤖 AI Market Forecast for Tomorrow")

    with st.spinner("Running neural network sequence..."):
        features = ["ATH_Proximity", "ATL_Proximity", "Intraday_Trend", "RSI_14", "SMA_20"]
        latest_features = df[features].tail(1).values
        scaled_features = scaler.transform(latest_features)
        predictions = model.predict(scaled_features)

        dir_prob = float(predictions[0][0][0])
        ret_low = float(predictions[1][0][0])
        ret_expected = float(predictions[2][0][0])
        ret_high = float(predictions[3][0][0])

        last_close = float(df["Close"].iloc[-1])

        price_expected = last_close * (1 + ret_expected)
        price_low = last_close * (1 + ret_low)
        price_high = last_close * (1 + ret_high)

    trend = "UP 📈" if dir_prob > 0.5 else "DOWN 📉"
    confidence = dir_prob if dir_prob > 0.5 else (1 - dir_prob)
    confidence_color = "normal" if confidence > 0.6 else "off"

    st.markdown("#### Network Directional Bias")
    col_cur, col_dir, col_conf = st.columns(3)
    col_cur.metric("Current Price", f"₹{last_close:.2f}")
    col_dir.metric("Predicted Direction", trend)
    col_conf.metric("AI Confidence", f"{confidence * 100:.2f}%", delta_color=confidence_color)

    st.markdown("<hr style='margin-top: 1rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.markdown("#### Target Volatility Range")
    
    col_floor, col_target, col_ceiling = st.columns(3)

    with col_floor:
        st.error(f"**Worst Case (Floor)**\n\n## ₹{price_low:.2f}")

    with col_target:
        st.info(f"**Expected Price**\n\n## ₹{price_expected:.2f}")

    with col_ceiling:
        st.success(f"**Best Case (Ceiling)**\n\n## ₹{price_high:.2f}")

with st.expander("🛠️ View Raw Data & Developer Debug Info"):
    st.dataframe(df.tail(5), use_container_width=True)
    st.write("**Model Input Shape:**", model.input_shape)
    st.write("**Raw Input Array:**", latest_features)
    st.write("**Scaled Input Array:**", scaled_features)