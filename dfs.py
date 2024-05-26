import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

# Streamlit app configuration
st.title('Stock Price Analysis')
st.sidebar.header('User Input')

# User input for stock symbol and time interval
stock_input = st.sidebar.text_input("Enter Stock", "AAPL")
time_input = st.sidebar.selectbox("Select Time", ['1D', '5D', '1M', '3M', '6M', '1Y', '5Y'])
avg_input = st.sidebar.selectbox("Select Days",[9,20,44,50,100,200])

# Define a dictionary to map time intervals to timedelta objects
time_interval_mapping = {
    '1D': timedelta(days=1),
    '5D': timedelta(days=5),
    '1M': timedelta(days=30),  # Assuming 1 month is approximately 30 days
    '3M': timedelta(days=90),
    '6M': timedelta(days=180),
    '1Y': timedelta(days=365),  # Assuming 1 year is approximately 365 days
    '5Y': timedelta(days=5 * 365),  # Assuming 5 years is approximately 5 * 365 days
}

# Get the timedelta object for the selected time interval
selected_time_interval = time_interval_mapping.get(time_input, timedelta(days=1))  # Default to 1 day if not found

# Calculate the start_date by subtracting the selected time interval from today's date
today_date = datetime.now().date()
start_date = today_date - selected_time_interval

# Format the start_date as a string in 'YYYY-MM-DD' format
start_date_str = start_date.strftime('%Y-%m-%d')

starting_date = start_date_str
end_date = today_date

st.write('Your start date is', starting_date)
st.write('Today\'s date is', today_date)

# Download stock data for the selected symbol and time interval
df = yf.download(stock_input, start=start_date, end=today_date)

moving_average_period = avg_input  # Change to your desired moving average period

# Download historical stock data using yfinance
stock_data = yf.download(stock_input, start=start_date, end=end_date)

# Calculate Simple Moving Average (SMA)
stock_data['SMA'] = stock_data['Close'].rolling(window=moving_average_period).mean()

# Calculate Exponential Moving Average (EMA)
stock_data['EMA'] = stock_data['Close'].ewm(span=moving_average_period, adjust=False).mean()


# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Customize date formatting
date_format = DateFormatter("%Y-%m-%d")

# Plotting the stock data
ax.plot(df.index, stock_data['EMA'], label=f'EMA ({time_input})', color='blue')
ax.plot(df.index, df['Close'], label=f'Close ({time_input})', color='red')
ax.plot(df.index, stock_data['SMA'], label=f'SMA ({time_input})', color='black')

# Adding labels and title
ax.set_ylabel('Price')
ax.set_title(f'Stock Price Over Time ({time_input})')

# Customize date formatting for the x-axis
ax.xaxis.set_major_formatter(date_format)

# Adding a legend
ax.legend()

# Customize the x-axis label
ax.set_xlabel('Date')

# Display the plot using Streamlit
st.pyplot(fig)