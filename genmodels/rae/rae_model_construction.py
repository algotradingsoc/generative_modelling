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
  def __init__(self, time_series_len, input_dim, LSTM_units, compression_units, regularization_coefficient):
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
    self.regularizer = tf.keras.regularizers.L1(regularization_coefficient)

    self.lss = LSTM_Sequential_State(time_series_len, input_dim, LSTM_units)
    self.d = tf.keras.layers.Dense(compression_units, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
    self.FF1 = tf.keras.layers.TimeDistributed(self.d)

    self.lss_scale = LSTM_Sequential_State(time_series_len, compression_units, LSTM_units)
    self.scale_up = tf.keras.layers.Dense(input_dim, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
    self.FF2 = tf.keras.layers.TimeDistributed(self.scale_up)

  def call(self, inputs):
    x = self.lss(inputs) # using default, ouputs: (1, 90, 10) 10: 2*5, where 5 is LSTM units
    x = self.FF1(x) # using default, ouputs: (1, 90, 1) -> latent time series (univariate)
    x = self.lss_scale(x) # check the dimension of x here
    x = self.FF2(x)
    return x

  def get_encoder(self):
    return lambda inputs : self.FF1(self.lss(inputs))

class RAE_Portfolio(tf.keras.Model):
  def __init__(self, time_series_len, input_dim, LSTM_units, compression_units, regularization_coefficient):
    '''
    time_series_len: the length of time series, which is the second dimension of the dataset
    input_dim: the number of time series, which is the third dimension of the dataset
    LSTM_units: the number of units used in LSTM layer
    compression_units: the number of units in the Dense layer which acts on each output at each time point

    output: a long-only no-leverage portfolio for input_dim stocks
    '''
    super(RAE_Portfolio, self).__init__()
    self.time_series_len = time_series_len
    self.input_dim = input_dim
    self.LSTM_units = LSTM_units
    self.compression_units = compression_units
    self.regularizer = tf.keras.regularizers.L1(regularization_coefficient)

    self.lss = LSTM_Sequential_State(time_series_len, input_dim, LSTM_units)
    self.d = tf.keras.layers.Dense(compression_units, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
    self.FF1 = tf.keras.layers.TimeDistributed(self.d)

    self.lss_scale = LSTM_Sequential_State(time_series_len, compression_units, LSTM_units)
    self.scale_up = tf.keras.layers.Dense(input_dim, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, activation="softmax")
    # softmax as activation guarantees long-only and no-leverage
    self.FF2 = tf.keras.layers.TimeDistributed(self.scale_up)

  def call(self, inputs):
    x = self.lss(inputs) # using default, ouputs: (1, 90, 10) 10: 2*5, where 5 is LSTM units
    x = self.FF1(x) # using default, ouputs: (1, 90, 1) -> latent time series (univariate)
    x = self.lss_scale(x) # check the dimension of x here
    x = self.FF2(x)
    return x

  def get_encoder(self):
    return lambda inputs : self.FF1(self.lss(inputs))
