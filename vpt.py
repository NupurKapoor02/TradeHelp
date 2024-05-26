import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Close': [100, 105, 110, 115, 120],
    'Volume': [10000, 12000, 15000, 11000, 9000]
}

df = pd.DataFrame(data)

vpt_values = []
vpt = 0
for i in range(1, len(df)):
    price_change = (df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1]
    vpt += price_change * df['Volume'].iloc[i]
    vpt_values.append(vpt)

df['VPT'] = [0] + vpt_values

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['VPT'], marker='o', linestyle='-', color='b')
plt.title('Volume Price Trend (VPT)')
plt.xlabel('Date')
plt.ylabel('VPT Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()