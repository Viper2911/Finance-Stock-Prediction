import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Market Forecast", layout="wide")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('📈 Upgraded Market Forecast App')
st.write("Powered by Prophet Machine Learning + Technical Indicators")

stocks = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
    'JPM', 'V', 'JNJ', 'WMT', 
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
    'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS'
)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_days = st.slider('Days of prediction into the future:', 1, 30, 7)

@st.cache_data
def get_exchange_rate():
    try:
        return yf.Ticker("INR=X").history(period="1d")['Close'].iloc[0]
    except:
        return 83.50 

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.reset_index(inplace=True)
    
    if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
        inr_rate = get_exchange_rate()
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in data.columns:
                data[col] = data[col] * inr_rate
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    change = data["Close"].diff()
    gain = change.mask(change < 0, 0.0)
    loss = -change.mask(change > 0, 0.0)
    rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-10)
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

data_load_state = st.text('Loading and engineering data...')
data = load_data(selected_stock)
data_load_state.text('Data loaded successfully! ✅')

st.subheader('Recent Market Data (Converted to ₹ INR)')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name="20-Day SMA", line=dict(color='orange', dash='dot')))
    fig.layout.update(title_text='Historical Time Series with Range Slider (Prices in ₹)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

df_train = data[['Date', 'Close', 'SMA_20', 'RSI_14']].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

with st.spinner('Training Prophet Model on Live Data...'):
    m = Prophet(daily_seasonality=True)
    
    m.add_regressor('SMA_20')
    m.add_regressor('RSI_14')
    
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=n_days)
    
    last_sma = df_train['SMA_20'].iloc[-1]
    last_rsi = df_train['RSI_14'].iloc[-1]
    
    future['SMA_20'] = last_sma
    future['RSI_14'] = last_rsi
    
    forecast = m.predict(future)

st.subheader('🤖 AI Forecast Data & Predicted Ranges')

future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days)
st.write(future_forecast)

st.info(f"**Target Range for Tomorrow:** The model expects the price to be roughly **₹{future_forecast['yhat'].iloc[0]:.2f}**, "
        f"but it could fluctuate anywhere between **₹{future_forecast['yhat_lower'].iloc[0]:.2f}** (Worst Case) "
        f"and **₹{future_forecast['yhat_upper'].iloc[0]:.2f}** (Best Case).")

st.write(f'Interactive Forecast plot for next {n_days} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.write("Forecast Components (Trends & Seasonality)")
fig2 = m.plot_components(forecast)
st.write(fig2)