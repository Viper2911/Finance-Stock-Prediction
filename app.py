import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Market Forecast",layout="wide")

START="2015-01-01"
TODAY=date.today().strftime("%Y-%m-%d")

st.title('Upgraded Market Forecast App')
st.write("Powered by Prophet Machine Learning + Technical Indicators")

stocks = ('AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'RELIANCE.NS', 'TCS.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_days=st.slider('Days of prediction into the future:',1,30,7)

