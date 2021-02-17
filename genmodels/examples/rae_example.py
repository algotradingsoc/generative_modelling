import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import gc
gc.collect()
from genmodels.rae.rae_model.constuction import RAE_Model

#!pip install yfinance
import yfinance as yf

### Dataset Construction - 16 stocks
tickers = "AAPL MSFT MMM MRO CLX CAT JPM GS AXP DLTR AMD NFLX FCX AZO KO AMZN"
data = yf.download(tickers, start="2017-01-01", end="2018-12-31")
## ignoring the fact it isnt div adjusted
rets = data["Adj Close"].apply(np.log).diff()[1:]

### Train Test Split
train_length = 252
train_rets = rets.values[:train_length, :]
test_rets = rets.values[train_length:, :]
def create_x_data(rets, time_series_len):
    num_time_series = len(rets) - time_series_len + 1
    x_data = []
    for i in range(num_time_series):
        x_data.append(rets[i:time_series_len+i,:input_dim])
    x_data = np.asarray(x_data, dtype=np.float32)
    return x_data
train_x_data = create_x_data(train_rets, time_series_len)
test_x_data = create_x_data(test_rets, time_series_len)

np.random.seed(42)
time_series_len = 40
LSTM_units = 5
input_dim = 16
compression_units = 4
regularization_coefficient = 2e-3
rae = RAE_Model(time_series_len=time_series_len, LSTM_units=LSTM_units, input_dim=input_dim, compression_units=compression_units, regularization_coefficient=regularization_coefficient)
rae.compile(optimizer="adam", loss="mse")
h = rae.fit(train_x_data, train_x_data, epochs=200, verbose=1, validation_data=(test_x_data, test_x_data), batch_size=10)

# Plot the loss function of the Generative Model (RAE)
fig = plt.figure(figsize=(12,7))
ax = plt.subplot(111)
ax.plot(h.history["loss"])
ax.plot(h.history["val_loss"])
ax.set_ylim((0, 0.001))
plt.show()

### Ex-Ante Forecasts

risk_aversion = 0.05
def negative_portfolio_utility(y_true,y_pred):
    # y_pred is the portfolio weights
    # y_true is the returns!
    print(y_pred.shape)
    print(y_true.shape)
    # we want to compute average portfolio return
    # mean-variance utility! similar to an exponential utility with the same risk aversion
    return -K.mean(K.sum(K.sum(y_pred * y_true, axis=-1), axis=-1)) + risk_aversion * K.mean(K.std(K.sum(y_pred * y_true, axis=-1), axis=-1))
    #return (callprice + kb.sum(y_pred * y_true,axis=-1) - kb.maximum(S_0 + kb.sum(y_true,axis=-1) - K,0.))**2
def create_future_returns(rets, num_time_series_len):
    future_returns = []
    for i in range(num_time_series_len):
        future_returns.append(rets[i+1:time_series_len+i+1,:input_dim])
    future_returns = np.asarray(future_returns, dtype=np.float32)
    return future_returns
  

# decompress to predict future returns!
train_encoded_components = rae.get_encoder()(train_x_data)[:-1,:,:]
train_encoded_components = tf.make_ndarray(tf.make_tensor_proto(train_encoded_components))
test_encoded_components = rae.get_encoder()(test_x_data)[:-1,:,:]
test_encoded_components = tf.make_ndarray(tf.make_tensor_proto(test_encoded_components))
train_future_returns = create_future_returns(train_rets, len(train_encoded_components)) * 100
test_future_returns = create_future_returns(test_rets, len(test_encoded_components)) * 100
train_encoded_components.shape, test_encoded_components.shape

l1_regularization_weight = 10.0
regularizer = tf.keras.regularizers.l1(l1_regularization_weight)

number_units = 8
model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(number_units, input_shape=(compression_units,), kernel_regularizer=regularizer, bias_regularizer=regularizer, activation='relu'))
model.add(tf.keras.layers.Dense(input_dim, kernel_regularizer=regularizer, bias_regularizer=regularizer, activation='tanh'))

optimizer=tf.keras.optimizers.Adam(lr=0.001)
epochs = 200
batch_size = 1
verbose = 1
model.compile(loss=negative_portfolio_utility, optimizer=optimizer)
prediction_history = model.fit(train_encoded_components, train_future_returns, validation_data=(test_encoded_components, test_future_returns), epochs=epochs, batch_size=batch_size, verbose=verbose)

# Plot the negative portfolio utility function of the Predictive Model built on Generative Model (RAE)
fig = plt.figure(figsize=(16,9))
ax = plt.subplot(111)
ax.plot(prediction_history.history["loss"], label="Training Loss")
ax.plot(prediction_history.history["val_loss"], label="Validation Loss")
ax.legend()
ax.set_ylim((-0.1,1.5))
plt.show()

# Compute the optimal portfolios and evaluation performance
train_portfolio_weights = model.predict(train_encoded_components)
test_portfolio_weights = model.predict(test_encoded_components)
train_portfolio_return_averages = np.mean(np.sum(train_portfolio_weights * train_future_returns, axis=-1), axis=-1)
train_portfolio_return_stds =  np.std(np.sum(train_portfolio_weights * train_future_returns, axis=-1), axis=-1)
test_portfolio_return_averages = np.mean(np.sum(test_portfolio_weights * test_future_returns, axis=-1), axis=-1)
test_portfolio_return_stds =  np.std(np.sum(test_portfolio_weights * test_future_returns, axis=-1), axis=-1)
