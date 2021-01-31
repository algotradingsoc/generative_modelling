"""
Recurrent Auto-Encoder
Original paper: https://arxiv.org/abs/1707.07961
"""

import tensorflow as tf
import keras.backend as K

### Model Creation

class LSTM_Sequential_State(tf.keras.Model):
  def __init__(self, time_series_len, input_dim, LSTM_units):
    '''
    This is a rolling version of LSTM, 
    which applies the ordinary LSTM to a dataset 
    and returns all the cell states and hidden states.
    '''
    super(LSTM_Sequential_State, self).__init__()
    self.time_series_len = time_series_len
    self.input_dim = input_dim
    self.LSTM_units = LSTM_units

    self.inputs = tf.keras.Input((self.time_series_len, self.input_dim))
    self.rnn = tf.keras.layers.LSTM(self.LSTM_units, return_state=True)

    def expand(x):
        return K.expand_dims(x, 1)

    self.expand_layer = tf.keras.layers.Lambda(expand, output_shape=lambda s: (s[0], 1, s[1]))

  @staticmethod
  def get_indexer(t):
        return tf.keras.layers.Lambda(lambda x, t: x[:, t, :], arguments={'t':t}, output_shape=lambda s: (s[0], s[2]))

  def call(self, inputs):
    state = None
    states = [] # list of (h, c) tuples
    outputs = []
    h_states = []
    c_states = []
    for t in range(self.time_series_len):
        input_t = LSTM_Sequential_State.get_indexer(t)(inputs)  # basically input_t = inputs[:, t, :]
        input_t = self.expand_layer(input_t)
        output_t, h, c = self.rnn(input_t, initial_state=state)
        state = h, c
        states.append(state)
        h_states.append(h)
        c_states.append(c)
        outputs.append(output_t)

    c_states_layer = tf.keras.layers.Concatenate(axis=-1)(c_states)
    c_reshape_layer = tf.keras.layers.Reshape((self.time_series_len,self.LSTM_units))(c_states_layer)
    h_states_layer = tf.keras.layers.Concatenate(axis=-1)(h_states)
    h_reshape_layer = tf.keras.layers.Reshape((self.time_series_len,self.LSTM_units))(h_states_layer)
    output = tf.keras.layers.concatenate((c_reshape_layer, h_reshape_layer), axis=-1)
    return output

class RAE_Model(tf.keras.Model):
  def __init__(self, time_series_len, input_dim, LSTM_units, compression_units):
    '''
    time_series_len: the length of time series, which is the first dimension of the dataset
    input_dim: the number of time series, which is the second dimension of the dataset
    LSTM_units: the number of units used in LSTM layer
    compression_units: the number of units in the Dense layer which acts on each output at each time point
    '''
    super(RAE_Model, self).__init__()
    self.time_series_len = time_series_len
    self.input_dim = input_dim
    self.LSTM_units = LSTM_units
    self.compression_units = compression_units

    self.lss = LSTM_Sequential_State(time_series_len, input_dim, LSTM_units)
    self.d = tf.keras.layers.Dense(compression_units)
    self.FF1 = tf.keras.layers.TimeDistributed(self.d)

    self.lss_scale = LSTM_Sequential_State(time_series_len, compression_units, LSTM_units)
    self.scale_up = tf.keras.layers.Dense(input_dim)
    self.FF2 = tf.keras.layers.TimeDistributed(self.scale_up)

  def call(self, inputs):
    x = self.lss(inputs) # using default, ouputs: (1, 90, 10) 10: 2*5, where 5 is LSTM units
    x = self.FF1(x) # using default, ouputs: (1, 90, 1) -> latent time series (univariate)
    x = self.lss_scale(x) # check the dimension of x here
    x = self.FF2(x)
    return x
    # freeze until one of the layers, and use the prior layers together with a feed-forward for forecasting future time series

  def get_encoder(self):
    return lambda inputs : self.FF1(self.lss(inputs))
    
tickers = "AAPL MSFT MMM MRO CLX CAT JPM GS AXP DLTR AMD NFLX FCX AZO TIF KO"
data = yf.download(tickers, period="100d")
## ignoring the fact it isnt div adjusted
rets = data["Adj Close"].apply(np.log).diff()[1:]
rets /= rets.rolling(10, min_periods=10).std()
rets = rets[10:]

time_series_len = 50
LSTM_units = 5
input_dim = 16
compression_units = 4
rae = RAE_Model(time_series_len=time_series_len, LSTM_units=LSTM_units, input_dim=input_dim, compression_units=compression_units)

rae.compile(optimizer="adam", loss="mse")
x = rets.values
x_data = np.asarray([x[:time_series_len,:input_dim]], dtype=np.float32)
h = rae.fit(x_data, x_data, epochs=2000, verbose=1)

# decompress to predict future returns!
current_encoded_components = rae.get_encoder()(x_data)[0,:,:]
current_encoded_components = tf.make_ndarray(tf.make_tensor_proto(current_encoded_components))
future_returns = rets.values[1:time_series_len+1,:input_dim]
train_ratio = 0.8
train_length = int(train_ratio*time_series_len)
#train_indices = np.random.choice(list(range(time_series_len)), int(train_ratio*time_series_len))
x_train = current_encoded_components[:train_length,:]
y_train = future_returns[:train_length,:]
x_validation = current_encoded_components[train_length:,:]
y_validation = future_returns[train_length:,:]

l1_regularization_weight = 4e-2
kernel_regularizer = tf.keras.regularizers.l1(l1_regularization_weight)

#number_units = 100
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(number_units, input_shape=(compression_units,), kernel_regularizer=kernel_regularizer, activation='relu'))
#model.add(tf.keras.layers.Dense(number_units, activation='relu'))
model.add(tf.keras.layers.Dense(input_dim, activation='relu'))

loss="mse"
optimizer=tf.keras.optimizers.Adam(lr=0.001)
metrics=['mae']
epochs = 2000
batch_size = 10
verbose = 0
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
prediction_history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size, verbose=verbose)

plt.plot(prediction_history.history["loss"])
plt.plot(prediction_history.history["val_loss"])

print("Prediction MSE:", prediction_history.history["loss"][-1], prediction_history.history["val_loss"][-1])
print("Total sum of squares are :", np.mean((y_train - np.mean(y_train))**2), np.mean((y_validation - np.mean(y_validation))**2))
