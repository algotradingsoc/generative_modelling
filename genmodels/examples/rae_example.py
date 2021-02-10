import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import gc
gc.collect()
from genmodels.rae.rae_model.constuction import RAE_Model

#!pip install yfinance
import yfinance as yf

### Dataset Construction

msft = yf.Ticker("MSFT")
hist = msft.history(period="100d")
## ignoring the fact it isnt div adjusted
rets = hist['Close'].apply(np.log).diff()
rets /= rets.rolling(10, min_periods=10).std()

aapl = yf.Ticker("AAPL")
hist = aapl.history(period="100d")
aapl_rets = hist['Close'].apply(np.log).diff()
aapl_rets /= aapl_rets.rolling(10, min_periods=10).std()
x = np.asarray([rets, aapl_rets]).T
x = x[10:]

time_series_len = 60
LSTM_units = 5
input_dim = 2
compression_units = 1
rae = RAE_Model(time_series_len=time_series_len, input_dim=input_dim, LSTM_units=LSTM_units, compression_units=compression_units)
rae.compile(optimizer="adam", loss="mse")
x_data = np.asarray([x[:time_series_len,:]], dtype=np.float32)
h = rae.fit(x_data, x_data, epochs=5000, verbose=0)

loss = h.history["loss"]

figsize = (16,9)
fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
plt.plot(loss, label="TS Len=60, units=5")
plt.xticks(fontsize=16)
plt.xlabel("epochs", fontsize=18)
plt.ylabel("", fontsize=18)
plt.title("Loss", fontsize=20)
ax.legend(fontsize=18)
filepath = u"/"
filename = u"RAE P.pdf"
plt.savefig()
