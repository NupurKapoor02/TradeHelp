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
stock_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL")
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
stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

def MACD():

    st.write("Moving Average Convergence Divergence")
    data = yf.download(stock_symbols, start=start_date, end=end_date)

    # Calculate the 12-period and 26-period exponential moving averages (EMAs)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    data['MACD'] = data['EMA12'] - data['EMA26']
    # Calculate the 9-period signal line
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Plot the MACD and Signal Line
    st.subheader(f'{stock_symbols}')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['MACD'], label=stock_symbols)
    ax.plot(data.index, data['Signal_Line'], label=f'{stock_symbols} Signal Line', color='black')
    ax.set_title(f'MACD for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('MACD Value')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks
    plt.xticks(rotation=45)
    ax.legend()

    # Generate buy and sell signals based on MACD
    data['Buy_Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)
    data['Sell_Signal'] = np.where(data['MACD'] < data['Signal_Line'], -1, 0)

    # Plot buy/sell signals
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == -1]

    ax.scatter(buy_signals.index, buy_signals['MACD'], marker='^', color='green', label='Buy Signal')
    ax.scatter(sell_signals.index, sell_signals['MACD'], marker='v', color='red', label='Sell Signal')

    st.pyplot(fig)

    # Display buy and sell options
    st.subheader("Buy Signals")
    st.dataframe(buy_signals)

    st.subheader("Sell Signals")
    st.dataframe(sell_signals)

MACD()