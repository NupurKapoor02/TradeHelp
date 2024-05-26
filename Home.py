import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

stock_input = st.sidebar.text_input("Enter Stock", "FEDERALBNK")
time_input = st.sidebar.selectbox("Select Time",['1D','5D','1M','3M','6M','1Y','5Y'])

symbol=stock_input

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

st.write('your start date is',starting_date)
st.write('todays date is',today_date)


dfs = [yf.download(symbol, period=interval) for interval in time_input]

# Create subplots with shared x-axis
fig, axes = plt.subplots(len(time_input), 1, figsize=(12, 6 * len(time_input)), sharex=True)

# Customize date formatting
date_format = DateFormatter("%Y-%m-%d")

# Plot each time interval on a separate subplot
for i, interval in enumerate(time_input):
    ax = axes[i]
    df = dfs[i]
    
    # Extract the date and closing price columns
    dates = df.index
    close_prices = df['Close']
    
    # Plotting the stock data
    ax.plot(dates, close_prices, label=f'Stock Price ({interval})', color='blue')
    
    # Adding labels and title
    ax.set_ylabel('Price')
    ax.set_title(f'Stock Price Over Time ({interval})')
    
    # Customize date formatting for the x-axis
    ax.xaxis.set_major_formatter(date_format)

    # Adding a legend
    ax.legend()

# Customize the x-axis label for the bottom subplot
axes[-1].set_xlabel('Date')

# Adjust layout spacing
plt.tight_layout()

# Display the plot
plt.show()