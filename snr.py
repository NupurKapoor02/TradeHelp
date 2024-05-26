import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import mplfinance as mpf

df = yf.download('^GSPC', start='2023-01-01')

print(df)

# getting support values using rolling function

support = df[df.Low == df.Low.rolling(5,center=True).min()].Low

resistance = df[df.High == df.High.rolling(5,center=True).max()].High

levels = pd.concat([support,resistance])

levels = levels[abs(levels.diff() > 100)]

mpf.plot(df,type='candle', hlines=levels.to_list(),style='charles')