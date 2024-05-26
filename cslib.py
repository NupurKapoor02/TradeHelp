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
def vpt():
    stock_data = {}  # Create a dictionary to store data for each symbol
    buy_threshold = 0  # Adjust the buy/sell thresholds as needed
    sell_threshold = 0
    
    for symbol in stock_symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        df['VPT'] = 0
        vpt_values = [0]
        
        for i in range(1, len(df)):
            price_change = (df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1]
            vpt = vpt_values[-1] + price_change * df['Volume'].iloc[i]
            vpt_values.append(vpt)
        
        df['VPT'] = vpt_values
        stock_data[symbol] = df
    
    st.header('Volume Price Trend (VPT) Analysis')
    
    for symbol in stock_symbols:
        st.subheader(f'{symbol}')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data[symbol].index, stock_data[symbol]['VPT'], label=symbol)
        ax.set_title(f'Volume Price Trend (VPT) for {symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('VPT Value')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks
        plt.xticks(rotation=45)
        
        # Generate buy/sell signals based on VPT values
        stock_data[symbol]['Buy_Signal'] = stock_data[symbol]['VPT'] > buy_threshold
        stock_data[symbol]['Sell_Signal'] = stock_data[symbol]['VPT'] < sell_threshold
        
        # Plot buy/sell signals
        ax.plot(stock_data[symbol].index[stock_data[symbol]['Buy_Signal']], stock_data[symbol]['VPT'][stock_data[symbol]['Buy_Signal']], '^', markersize=8, color='g', label='Buy Signal')
        ax.plot(stock_data[symbol].index[stock_data[symbol]['Sell_Signal']], stock_data[symbol]['VPT'][stock_data[symbol]['Sell_Signal']], 'v', markersize=8, color='r', label='Sell Signal')
        
        ax.legend()
        st.pyplot(fig)
        
        # Generate final buy/sell options
        buy_options = stock_data[symbol][stock_data[symbol]['Buy_Signal']]
        sell_options = stock_data[symbol][stock_data[symbol]['Sell_Signal']]
        
        if not buy_options.empty:
            st.subheader(f"Buy Options for {symbol}")
            st.dataframe(buy_options)
        
        if not sell_options.empty:
            st.subheader(f"Sell Options for {symbol}")
            st.dataframe(sell_options)

vpt()