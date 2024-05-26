import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import numpy as np
import pickle
import joblib

st.sidebar.header('User Input')
time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])

buysell=[]
final={} 
valu=[0,0,0,0,0,0,0,0,0]

large_avg_input = st.sidebar.text_input("Enter Large Moving Average", "50")
small_avg_input = st.sidebar.text_input("Enter Small Moving Average", "20")
large_avg_input=int(large_avg_input)
small_avg_input=int(small_avg_input)

# User input for stock symbols and date range
stock_symbols = st.sidebar.selectbox("Enter Stock Symbols (comma-separated)", ["AAPL","SBI","GOOGL","TSLA","INFY","TATASTEEL.NS"])
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
st.title('STOCK Analysis '+ str(stock_symbols[0]))
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
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data[symbol].index, stock_data[symbol]['VPT'], label=symbol)
        ax.set_title(f'Volume Price Trend (VPT) for {symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('VPT Value')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks
        plt.xticks(rotation=45)
        final['vpt']=df.copy()
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

        last_signal = ""
    if not df.empty:
        last_signal = "Buy" if df['Buy_Signal'].iloc[-1] == 1 else "Sell" if df['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("VPT: "+last_signal)
    if last_signal=='Buy':
        valu[7]=1
    elif last_signal=='Sell':
        valu[7]=-1
    else:
        valu[7]=0
    


def moving():
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

    moving_average_period = avg_input
    st.header('Moving Avrages Analysis')
    stock_data['SMA'] = stock_data['Close'].rolling(window=large_avg_input).mean()
    stock_data['EMA'] = stock_data['Close'].ewm(span=small_avg_input, adjust=False).mean()

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
    final['moving']=stock_data.copy()
    # Generate Buy/Sell signals
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

    # Display the final recommendation at the end of the plot
    st.pyplot(fig)
    
    # Display Buy/Sell options
    st.subheader("Buy Signals (SMA > EMA)")
    st.dataframe(buy_points[['Close', 'SMA', 'EMA']])

    st.subheader("Sell Signals (SMA < EMA)")
    st.dataframe(sell_points[['Close', 'SMA', 'EMA']])

    # Calculate the final recommendation based on the last data point for the entire time range
    last_signal = ""
    if not stock_data.empty:
        last_signal = "Buy" if stock_data['Buy_Signal'].iloc[-1] == 1 else "Sell" if stock_data['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("Moving Avrages: "+last_signal)
    if last_signal=='Buy':
        valu[1]=1
        valu[2]=1
    elif last_signal=='Sell':
        valu[1]=-1
        valu[2]=-1
    else:
        valu[1]=0
        valu[2]=0
    

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
    st.write("RSI Index")
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

    rsi_period = avg_input
    stock_data['RSI'] = calculate_rsi(stock_data, period=rsi_period)
    st.header('RSI Analysis')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['RSI'], label='RSI')
    ax.plot(stock_data.index, stock_data['Close'], label='Close', color='black')
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax.set_title(f'Relative Strength Index (RSI) for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend()
    final['rsi']=stock_data.copy()
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

    last_buy_signal = stock_data['Buy_Signal'].iloc[-1]
    last_sell_signal = stock_data['Sell_Signal'].iloc[-1]
    last_signal = ""
    if not stock_data.empty:
        last_signal = "Buy" if stock_data['Buy_Signal'].iloc[-1] == 1 else "Sell" if stock_data['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("RSI: "+last_signal)
    if last_signal=='Buy':
        valu[0]=1
    elif last_signal=='Sell':
        valu[0]=-1
    else:
        valu[0]=0


def calculate_mfi(data, period):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    
    positive_flow = (raw_money_flow.where(data['Close'] > data['Close'].shift(1), 0)).rolling(window=period).sum()
    negative_flow = (raw_money_flow.where(data['Close'] < data['Close'].shift(1), 0)).rolling(window=period).sum()
    
    money_flow_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi
def mfi():

    st.write('Money Flow Index')
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    mfi_period = avg_input  # You can adjust this period as needed
    stock_data['MFI'] = calculate_mfi(stock_data, period=mfi_period)
    st.header('MFI Analysis')
    final['mfi']=stock_data.copy()
    # Add buy/sell signals based on MFI values
    stock_data['Buy_Signal'] = (stock_data['MFI'] < 30) & (stock_data['MFI'].shift(1) >= 30)
    stock_data['Sell_Signal'] = (stock_data['MFI'] > 70) & (stock_data['MFI'].shift(1) <= 70)


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
        st.subheader("Buy Options (MFI < 30)")
        st.dataframe(buy_signals[['MFI']])

    if not sell_signals.empty:
        st.subheader("Sell Options (MFI > 70)")
        st.dataframe(sell_signals[['MFI']])

    last_signal = ""
    if not stock_data.empty:
        last_signal = "Buy" if stock_data['Buy_Signal'].iloc[-1] == 1 else "Sell" if stock_data['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("MFI: "+last_signal)
    if last_signal=='Buy':
        valu[4]=1
    elif last_signal=='Sell':
        valu[4]=-1
    else:
        valu[4]=0


def roc():
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    roc_period = avg_input # You can adjust this period as needed
    stock_data['ROC'] = ((stock_data['Close'] - stock_data['Close'].shift(roc_period)) / stock_data['Close'].shift(roc_period)) * 100
    st.header('ROC Analysis')
    final['roc']=stock_data.copy()
    # Add buy/sell signals based on ROC values
    threshold = 0  # You can adjust this threshold as needed
    stock_data['Buy_Signal'] = stock_data['ROC'] < threshold
    stock_data['Sell_Signal'] = stock_data['ROC'] > threshold

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
        st.subheader("Buy Options (ROC < 0)")
        st.dataframe(buy_options)
    
    if not sell_options.empty:
        st.subheader("Sell Options (ROC > 0))")
        st.dataframe(sell_options)

    last_signal = ""
    if not stock_data.empty:
        last_signal = "Buy" if stock_data['Buy_Signal'].iloc[-1] == 1 else "Sell" if stock_data['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("ROC: "+last_signal)
    if last_signal=='Buy':
        valu[3]=1
    elif last_signal=='Sell':
        valu[3]=-1
    else:
        valu[3]=0


def bollinger():

    # Fetch historical data using yfinance
    df = yf.download(stock_symbols, start=start_date, end=end_date)
    st.header('Bollinger Bands Analysis')
    # Calculate Bollinger Bands
    period = 20
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['StdDev'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (df['StdDev'] * 2)
    df['Lower'] = df['SMA'] - (df['StdDev'] * 2)

    # Plot Bollinger Bands
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close', color='purple')
    plt.plot(df.index, df['Upper'], label='Upper Bollinger Band', color='red', linestyle='--')
    plt.plot(df.index, df['Lower'], label='Lower Bollinger Band', color='green', linestyle='--')
    plt.fill_between(df.index, df['Lower'], df['Upper'], alpha=0.2, color='yellow')
    ax.set_title(f'Bollinger Bands for {stock_symbols}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Bollinger Bands')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date ticks
    final['bollinger']=df.copy()
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
    st.subheader("Buy Signals Closing<Lower Bollinger Band")
    st.dataframe(buy_signals)

    st.subheader("Sell Signals Closing>Upper Bollinger Band")
    st.dataframe(sell_signals)

    last_signal = ""
    if not df.empty:
        last_signal = "Buy" if df['Buy_Signal'].iloc[-1] == 1 else "Sell" if df['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("Bollinger: "+last_signal)
    


def MACD():

    data = yf.download(stock_symbols, start=start_date, end=end_date)

    # Calculate the 12-period and 26-period exponential moving averages (EMAs)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    st.header('MACD Analysis')
    # Calculate the MACD line
    data['MACD'] = data['EMA12'] - data['EMA26']
    # Calculate the 9-period signal line
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

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
    final['macd']=data.copy()
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
    st.subheader("Buy Signals Closing>Signal Line")
    st.dataframe(buy_signals)

    st.subheader("Sell Signals Closing<Signal Line")
    st.dataframe(sell_signals)
    
    last_signal = ""
    if not data.empty:
        last_signal = "Buy" if data['Buy_Signal'].iloc[-1] == 1 else "Sell" if data['Sell_Signal'].iloc[-1] == -1 else "Hold"

    buysell.append("MACD: "+last_signal)
    if last_signal=='Buy':
        valu[5]=1
        valu[6]=1
    elif last_signal=='Sell':
        valu[5]=-1
        valu[6]=-1
    else:
        valu[5]=0
        valu[6]=0

MACD()
bollinger()
vpt()
moving()
rsi()
mfi()
roc()
st.header('Final Analysis') 
st.header(buysell)
final_df= pd.DataFrame()
final_df['RSI']=final['rsi']['RSI'].copy()
final_df['SMA_50']=final['moving']['SMA'].copy()
final_df['EMA_20']=final['moving']['EMA'].copy()
final_df['ROC']=final['roc']['ROC'].copy()
final_df['MFI']=final['mfi']['MFI'].copy()
final_df['Macd_EMA12']=final['macd']['EMA12'].copy()
final_df['Macd_EMA26']=final['macd']['EMA26'].copy()
final_df['VPT']=final['vpt']['VPT'].copy()


st.header('Prediction')
model = joblib.load("./XGBoost_Model2.pkl")  # Replace with your file path
importance_values = model.feature_importances_

# Optionally, get the names of the features if available
try:
    feature_names = model.get_booster().feature_names
except AttributeError:
    # If feature names are not available, you can use feature indices as names
    feature_names = [f'feature_{i}' for i in range(len(importance_values))]

# Create a dictionary to map feature names to their importance values
feature_importance_dict = dict(zip(feature_names, importance_values))

# Print or use the feature importance values
sum=0
flag=0
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")
    sum+=importance*valu[flag]
    flag+=1
st.write(sum)
if sum>0.2:
    st.header("Buy")
elif sum<-0.2:
    st.header("Sell")
else:
    st.header("Hold")
