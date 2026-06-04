import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objs as go
from textblob import TextBlob
import shap

st.set_page_config(page_title="Institutional Quant Engine", layout="wide", page_icon="🏦")
st.markdown("<h1 style='text-align: center;'>🏦 Institutional Quant Engine</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #888888; margin-bottom: 2rem;'>Multi-Modal Analytics: Technicals + Fundamentals + NLP Sentiment</h5>", unsafe_allow_html=True)

stocks = (
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS', 
    'BHARTIARTL.NS', 'SUNPHARMA.NS', 'LAURUSLABS.NS', 'DRREDDY.NS', 'NESTLEIND.NS', 
    'HINDUNILVR.NS', 'BEL.NS', 'IOC.NS', 'BAJFINANCE.NS', 'JIOFIN.NS', 'CDSL.NS',
    'ETERNAL.NS', 'VEDL.NS'
)

with st.container(border=True):
    selected_stock = st.selectbox("🔍 Search & Select Market Asset", stocks)

@st.cache_resource
def load_ml_assets():
    model = load_model("quantile_market_model.h5", compile=False)
    scaler = joblib.load("market_scaler.pkl")
    try:
        explainer = joblib.load("shap_explainer.pkl")
    except:
        explainer = None
    return model, scaler, explainer

def fetch_alternative_data(ticker_obj):
    news = ticker_obj.news
    sentiment_score = 0.0
    if news:
        sentiments = []
        for article in news[:5]:
            blob = TextBlob(article.get('title', ''))
            sentiments.append(blob.sentiment.polarity)
        sentiment_score = np.mean(sentiments) if sentiments else 0.0

    info = ticker_obj.info
    pe_ratio = info.get('trailingPE', 20.0)
    if pe_ratio is None: pe_ratio = 20.0
    valuation_score = 20.0 / pe_ratio if pe_ratio > 0 else 1.0

    return sentiment_score, valuation_score, pe_ratio

@st.cache_data(ttl=3600)
def load_and_engineer_data(ticker):
    ticker_obj = yf.Ticker(ticker)
    live_sentiment, live_valuation, raw_pe = fetch_alternative_data(ticker_obj)

    macro_tickers = {"NIFTY50": "^NSEI", "USD_INR": "INR=X", "SP500": "^GSPC", "CRUDE_OIL": "CL=F", "VIX": "^VIX"}
    macro_dfs = []
    for name, m_ticker in macro_tickers.items():
        try:
            m_df = yf.download(m_ticker, period="10y", interval="1d", progress=False)
            if not m_df.empty:
                if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.droplevel(1)
                m_df[f"Macro_{name}_Return"] = m_df["Close"].pct_change()
                macro_dfs.append(m_df[[f"Macro_{name}_Return"]])
        except: pass

    macro_master = pd.concat(macro_dfs, axis=1).ffill().bfill()
    df = yf.download(ticker, period="10y", interval="1d", progress=False)
    
    if df.empty:
        st.error("No data found.")
        st.stop()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

    df["ATH_Proximity"] = df["Close"] / df["High"].cummax()
    df["ATL_Proximity"] = df["Close"] / df["Low"].cummin()
    df["Intraday_Trend"] = (df["Close"] - df["Open"]) / df["Open"]
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Volume_Shock"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    
    df["News_Sentiment"] = live_sentiment
    df["Fundamental_Valuation"] = live_valuation

    df = df.join(macro_master, how="left").dropna()
    df.reset_index(inplace=True)
    if "Date" not in df.columns: df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    return df, live_sentiment, raw_pe

try:
    model, scaler, explainer = load_ml_assets()
except Exception as e:
    st.error(f"⚠️ Missing Model Files.")
    st.stop()

with st.spinner("Scraping alternative data and executing neural sequence..."):
    df, live_sentiment, raw_pe = load_and_engineer_data(selected_stock)

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
    st.subheader("🌐 Real-Life Factor Analysis")
    col_nlp, col_fund = st.columns(2)
    
    sent_label = "Bullish 🟢" if live_sentiment > 0.05 else ("Bearish 🔴" if live_sentiment < -0.05 else "Neutral ⚪")
    col_nlp.metric("Live NLP News Sentiment", f"{live_sentiment:.2f}", sent_label, delta_color="off")
    
    val_label = "Undervalued 🟢" if raw_pe < 20 else "Overvalued 🔴"
    col_fund.metric("Current P/E Ratio", f"{raw_pe:.1f}", val_label, delta_color="off")

with st.container(border=True):
    st.subheader("🤖 Multi-Modal Price Forecast")

    features = [
        "ATH_Proximity", "ATL_Proximity", "Intraday_Trend", "SMA_20", "Volume_Shock", "News_Sentiment", "Fundamental_Valuation",
        "Macro_NIFTY50_Return", "Macro_USD_INR_Return", "Macro_SP500_Return", "Macro_CRUDE_OIL_Return", "Macro_VIX_Return"
    ]
    
    latest_features = df[features].tail(10).values
    scaled_features = scaler.transform(latest_features)
    reshaped_features = scaled_features.reshape(1, 10, len(features))
    
    predictions = model.predict(reshaped_features)
    dir_prob = float(predictions[0][0][0])
    pred_low_return = float(predictions[1][0][0])
    pred_close_return = float(predictions[2][0][0])
    pred_high_return = float(predictions[3][0][0])

    last_close = float(df["Close"].iloc[-1])
    
    price_expected = last_close * (1 + pred_close_return)
    price_low = min(last_close * (1 + pred_low_return), price_expected)
    price_high = max(last_close * (1 + pred_high_return), price_expected)

    THRESHOLD = 0.50
    trend = "UP 📈" if dir_prob > THRESHOLD else "DOWN 📉"
    confidence = dir_prob if dir_prob > THRESHOLD else (1.0 - dir_prob)
    confidence_color = "normal" if confidence > 0.6 else "off"

    col_cur, col_dir, col_conf = st.columns(3)
    col_cur.metric("Current Price", f"₹{last_close:.2f}")
    col_dir.metric("Predicted Direction", trend)
    col_conf.metric("AI Confidence", f"{confidence * 100:.2f}%", delta_color=confidence_color)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col_floor, col_target, col_ceiling = st.columns(3)
    col_floor.error(f"**Worst Case (Floor)**\n\n## ₹{price_low:.2f}")
    col_target.info(f"**Expected Price**\n\n## ₹{price_expected:.2f}")
    col_ceiling.success(f"**Best Case (Ceiling)**\n\n## ₹{price_high:.2f}")

with st.expander("📊 View Raw Data", expanded=False):
    st.dataframe(df.tail(10), use_container_width=True)

with st.expander("🧠 AI Logic & Explainability", expanded=False):
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(reshaped_features)
            feature_importance = np.abs(shap_values[2][0]).sum(axis=0)
            
            fig_shap = go.Figure(go.Bar(
                x=feature_importance,
                y=features,
                orientation='h',
                marker=dict(color='#00ffcc')
            ))
            fig_shap.update_layout(title="Feature Importance for Tomorrow's Prediction", yaxis={'categoryorder':'total ascending'}, template="plotly_dark")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning("SHAP calculation currently initializing.")
    else:
        st.warning("shap_explainer.pkl not found.")

with st.expander("🛠️ Developer Debug Info", expanded=False):
    st.write("**Model Input Shape:**", model.input_shape)
    st.write("**Raw Input Array:**", latest_features)
    st.write("**Scaled Input Array:**", scaled_features)