import tensorflow as tf

"""
  Helper functions to build NNs
"""

def make_nn(output_size, hidden_sizes):
    """ 
      Creates a fully connected neural network. 
      Params:
        output_size: output dimensionality
        hidden_sizes: list of hidden layer sizes. List length is the number of hidden layers.
    """
    NNLayers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32) for h in hidden_sizes]
    NNLayers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(NNLayers)


def make_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of a 1D Conv layer (to use temporal relationships)
        followed by a fully connected network
        Params:
          output_size: output dimension
          hidden_sizes: list  of hidden layer sizes.
          kernel_size: kernel size for conv layer
    """
    CNN_layers = [tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size, padding="same", activation="relu", dtype=tf.float32)]
    
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32) for h in hidden_sizes[1:]]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    layers = CNN_layers + layers
    return tf.keras.Sequential(layers)
