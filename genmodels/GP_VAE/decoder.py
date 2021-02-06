import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from helpers import *

"""
    Decoder with (Multivariate) Gaussian prior
"""

class GaussianDecoder(tf.keras.Model):
    def __init__(self, output_size, hidden_sizes=(64, 64), paper_version=False):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes. Tuple length = number of hidden layers
            :param: paper_version: bool indicating whether to use the paper's verion of the decoder
                                        paper only uses a network for the mean and holds variance at 1 for all time series
        """
        super(GaussianDecoder, self).__init__()
        self.paper_version = paper_version
        self.network = make_nn(output_size, hidden_sizes)
        if not self.paper_version:
          self.network_logvar = make_nn(output_size, hidden_sizes)

    def __call__(self, x):
        mean = self.network(x)
        if self.paper_version:
          var = tf.ones(tf.shape(mean), dtype=tf.float32)
        else:
          var = tf.math.sqrt(tf.math.exp(0.5 * self.network_logvar(x)))

        return tfd.Normal(loc=mean, scale=var)