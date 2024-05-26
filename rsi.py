import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import numpy as np

# Define the stock symbol and time period
st.sidebar.header('User Input')
time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])

# User input for stock symbols and date range
stock_symbol = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL")
time_interval_mapping = {
    '1D': timedelta(days=1),
    '5D': timedelta(days=5),
    '1M': timedelta(days=30),  # Assuming 1 month is approximately 30 days
    '3M': timedelta(days=90),
    '6M': timedelta(days=180),
    '1Y': timedelta(days=365),  # Assuming 1 year is approximately 365 days
    '5Y': timedelta(days=5 * 365),  
}

selected_time_interval = time_interval_mapping.get(time_input, timedelta(days=1))
end_date = datetime.now().date()
start_date = end_date - selected_time_interval
# Download historical stock data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate RSI
def calculate_rsi(data, period):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def rsi():
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    rsi_period = avg_input
    stock_data['RSI'] = calculate_rsi(stock_data, period=rsi_period)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['RSI'], label='RSI')
    ax.plot(stock_data.index, stock_data['Close'], label='Close', color='black')
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax.set_title(f'Relative Strength Index (RSI) for {stock_symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend()

    # Calculate Buy/Sell signals based on RSI, considering values greater than 70 and less than 30
    stock_data['Buy_Signal'] = np.where(stock_data['RSI'] < 30, 1, 0)
    stock_data['Sell_Signal'] = np.where(stock_data['RSI'] > 70, -1, 0)

    # Plot Buy/Sell signals on the RSI plot
    buy_points = stock_data[stock_data['Buy_Signal'] == 1]
    sell_points = stock_data[stock_data['Sell_Signal'] == -1]

    ax.scatter(buy_points.index, buy_points['RSI'], marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(sell_points.index, sell_points['RSI'], marker='v', color='red', label='Sell Signal', alpha=1)

    st.pyplot(fig)

    # Display Buy/Sell options for RSI values greater than 70 or less than 30
    st.subheader("Buy Signals (RSI < 30)")
    st.dataframe(buy_points[buy_points['RSI'] < 30][['Close', 'RSI']])

    st.subheader("Sell Signals (RSI > 70)")
    st.dataframe(sell_points[sell_points['RSI'] > 70][['Close', 'RSI']])

rsi()