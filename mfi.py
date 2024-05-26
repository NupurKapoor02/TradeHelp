import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

# Define the stock symbol and time period
st.title('Volume Price Trend (VPT) Analysis')
st.sidebar.header('User Input')
time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])
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
# Download historical stock data using yfinance
def calculate_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    
    positive_flow = (raw_money_flow.where(data['Close'] > data['Close'].shift(1), 0)).rolling(window=period).sum()
    negative_flow = (raw_money_flow.where(data['Close'] < data['Close'].shift(1), 0)).rolling(window=period).sum()
    
    money_flow_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi
def mfi():
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    mfi_period = 14  # You can adjust this period as needed
    stock_data['MFI'] = calculate_mfi(stock_data, period=mfi_period)

    # Add buy/sell signals based on MFI values
    stock_data['Buy_Signal'] = (stock_data['MFI'] < 30) & (stock_data['MFI'].shift(1) >= 30)
    stock_data['Sell_Signal'] = (stock_data['MFI'] > 70) & (stock_data['MFI'].shift(1) <= 70)

    st.subheader(f'{stock_symbols}')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot MFI
    ax.plot(stock_data.index, stock_data['MFI'], label='MFI', color='purple')

    # Plot buy/sell signals using scatter plots
    buy_signals = stock_data[stock_data['Buy_Signal']]
    sell_signals = stock_data[stock_data['Sell_Signal']]
    
    ax.scatter(buy_signals.index, buy_signals['MFI'], marker='^', color='g', label='Buy Signal', s=100)
    ax.scatter(sell_signals.index, sell_signals['MFI'], marker='v', color='r', label='Sell Signal', s=100)

    ax.set_title(f'Money Flow Index (MFI) for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('MFI Value')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-d'))  # Format the date ticks
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Display buy/sell options
    if not buy_signals.empty:
        st.subheader("Buy Options")
        st.dataframe(buy_signals[['MFI']])

    if not sell_signals.empty:
        st.subheader("Sell Options")
        st.dataframe(sell_signals[['MFI']])

mfi()