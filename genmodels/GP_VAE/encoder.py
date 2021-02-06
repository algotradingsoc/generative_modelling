import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from helpers import *

"""
    The default encoder for time-series imputation
"""

class JointEncoder(tf.keras.Model):
    def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, cov_activation=tf.nn.softplus):
        """ Encoder with 1D conv network and Normal posterior with independent marginal distributions
            Used by GP-VAE with Normal posterior with independent marginals
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes. Tuple length = number of hidden layers
            :param window_size: kernel size for Conv layer
            :param cov_activation: function to apply to covariance matrix diagonals to ensure > 0? Authors use softplus.
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_cnn(2*z_size, hidden_sizes, window_size)
        self.cov_activation = cov_activation

    def __call__(self, x):
        mapped = self.net(x)
        n = len(x.shape.as_list()) # Num dims
        perm = list(range(n-2)) + [n-1, n-2]
        mapped = tf.transpose(mapped, perm=perm)
        loc = mapped[..., :self.z_size, :]
        scale_diag = self.cov_activation(mapped[..., self.z_size:, :])
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)