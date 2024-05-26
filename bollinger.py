import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import numpy as np

st.title('Volume Price Trend (VPT) Analysis')
st.sidebar.header('User Input')
time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])

large_avg_input = st.sidebar.text_input("Enter Large Moving Average", "50")
small_avg_input = st.sidebar.text_input("Enter Small Moving Average", "20")
large_avg_input=int(large_avg_input)
small_avg_input=int(small_avg_input)

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
stock_symbols = [symbol.strip() for symbol in stock_symbols.split(',')]
stock_data = {}
def bollinger():
    st.write('Bollinger Bands')

    # Fetch historical data using yfinance
    df = yf.download(stock_symbols, start=start_date, end=end_date)

    # Calculate Bollinger Bands
    period = 20
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['StdDev'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (df['StdDev'] * 2)
    df['Lower'] = df['SMA'] - (df['StdDev'] * 2)

    # Plot Bollinger Bands
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='MFI', color='purple')
    plt.plot(df.index, df['Upper'], label='Upper Bollinger Band', color='red', linestyle='--')
    plt.plot(df.index, df['Lower'], label='Lower Bollinger Band', color='green', linestyle='--')
    plt.fill_between(df.index, df['Lower'], df['Upper'], alpha=0.2, color='yellow')
    ax.set_title(f'Bollinger Bands for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Bollinger Bands')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks

    # Identify Buy and Sell signals based on Bollinger Bands
    df['Buy_Signal'] = np.where(df['Close'] < df['Lower'], 1, 0)
    df['Sell_Signal'] = np.where(df['Close'] > df['Upper'], -1, 0)

    # Plot Buy/Sell signals on the Bollinger Bands plot
    buy_signals = df[df['Buy_Signal'] == 1]
    sell_signals = df[df['Sell_Signal'] == -1]

    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')

    ax.legend()
    st.pyplot(fig)

    # Display Buy and Sell options
    st.subheader("Buy Signals")
    st.dataframe(buy_signals)

    st.subheader("Sell Signals")
    st.dataframe(sell_signals)

bollinger()