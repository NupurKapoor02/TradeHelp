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
def roc():
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    roc_period = 1  # You can adjust this period as needed
    stock_data['ROC'] = ((stock_data['Close'] - stock_data['Close'].shift(roc_period)) / stock_data['Close'].shift(roc_period)) * 100

    # Add buy/sell signals based on ROC values
    threshold = 0  # You can adjust this threshold as needed
    stock_data['Buy_Signal'] = stock_data['ROC'] > threshold
    stock_data['Sell_Signal'] = stock_data['ROC'] < threshold

    st.subheader(f'{stock_symbols}')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['ROC'], label=stock_symbols)
    ax.set_title(f'Rate of Change (ROC) for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('ROC Value')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks
    plt.xticks(rotation=45)
    
    if any(stock_data['Buy_Signal']):
        # Plot buy signals
        ax.plot(stock_data.index[stock_data['Buy_Signal']], stock_data['ROC'][stock_data['Buy_Signal']], '^', markersize=8, color='g', label='Buy Signal')
    if any(stock_data['Sell_Signal']):
        # Plot sell signals
        ax.plot(stock_data.index[stock_data['Sell_Signal']], stock_data['ROC'][stock_data['Sell_Signal']], 'v', markersize=8, color='r', label='Sell Signal')
    
    ax.legend()
    st.pyplot(fig)
    
    # Generate final buy/sell options
    buy_options = stock_data[stock_data['Buy_Signal']]
    sell_options = stock_data[stock_data['Sell_Signal']]
    
    if not buy_options.empty:
        st.subheader("Buy Options")
        st.dataframe(buy_options)
    
    if not sell_options.empty:
        st.subheader("Sell Options")
        st.dataframe(sell_options)

roc()