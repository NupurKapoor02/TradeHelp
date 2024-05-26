import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import numpy as np

time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])

large_avg_input = st.sidebar.text_input("Enter Large Moving Average", "50")
small_avg_input = st.sidebar.text_input("Enter Small Moving Average", "20")


large_avg_input = int(large_avg_input)
small_avg_input = int(small_avg_input)
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

def moving():
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

    moving_average_period = avg_input

    stock_data['SMA'] = stock_data['Close'].rolling(window=large_avg_input).mean()
    stock_data['EMA'] = stock_data['Close'].ewm(span=small_avg_input, adjust=False).mean()
    # stock_data['EMA'] = stock_data['Close'].ewm(span=moving_average_period, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    date_format = DateFormatter("%Y-%m-d")

    ax.plot(stock_data.index, stock_data['EMA'], label=f'EMA ({time_input})', color='blue')
    ax.plot(stock_data.index, stock_data['Close'], label=f'Close ({time_input})', color='red')
    ax.plot(stock_data.index, stock_data['SMA'], label=f'SMA ({time_input})', color='black')

    ax.set_ylabel('Price')
    ax.set_title(f'Stock Price Over Time ({time_input})')
    ax.xaxis.set_major_formatter(date_format)
    ax.legend()
    ax.set_xlabel('Date')

    stock_data['Buy_Signal'] = np.where(
    (stock_data['SMA'] > stock_data['EMA']) & 
    ((stock_data['Close'] >= stock_data['SMA'] - 0.1 * stock_data['SMA']) | (stock_data['Close'] <= stock_data['EMA'] + 0.1 * stock_data['EMA'])),
    1, 0)

    stock_data['Sell_Signal'] = np.where(
    (stock_data['SMA'] < stock_data['EMA']) & 
    ((stock_data['Close'] >= stock_data['EMA'] - 0.1 * stock_data['EMA']) | (stock_data['Close'] <= stock_data['SMA'] + 0.1 * stock_data['SMA'])),
    -1, 0)

    
    # Plot Buy/Sell signals
    buy_points = stock_data[stock_data['Buy_Signal'] == 1]
    sell_points = stock_data[stock_data['Sell_Signal'] == -1]

    ax.scatter(buy_points.index, buy_points['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(sell_points.index, sell_points['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

    st.pyplot(fig)

    # Calculate the final recommendation based on the last data point for the entire time range
    last_buy_signal = stock_data['Buy_Signal'].iloc[-1]
    last_sell_signal = stock_data['Sell_Signal'].iloc[-1]
    if last_buy_signal == 1:
        final_recommendation = 'Buy'
    elif last_sell_signal == -1:
        final_recommendation = 'Sell'
    else:
        final_recommendation = 'No Signal'

    # Display the final recommendation at the end of the plot
    st.write(f"Final Recommendation: {final_recommendation}")

# Call the moving function to display the plot and calculate the final recommendation
moving()