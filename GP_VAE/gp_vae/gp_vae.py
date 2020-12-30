import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
from collections import defaultdict
import time

from kernels import *
from encoder import *
from decoder import *

"""
    GP-VAE model
"""

# Required parameters when instantiating
required_params = ["latent_dim", "data_dim", "time_length"]


class GP_VAE(tf.keras.Model):
    def __init__(self, params):
        super(GP_VAE, self).__init__()

        #assert required_params in list(params.keys()), "Need to provide required parameters: " + ", ".join(np.setdiff(required_params, list(params.keys())))

        if "seed" in params.keys():
            np.random.seed(params["seed"])
            tf.random.set_seed(params["seed"])
        
        self.params = params
        self.initialise(latent_dim=params['latent_dim'], 
                        data_dim=params['data_dim'], 
                        time_length=params['time_length'], 
                        encoder_sizes=params['encoder_sizes'],
                        decoder_sizes=params['decoder_sizes'],
                        kernel=params['kernel'], 
                        sigma=params['sigma'],
                        length_scale=params['length_scale'],
                        kernel_scales=params['kernel_scales'],
                        beta=params['beta'],
                        M=params['M'],
                        K=params['K'],
                        window_size=params['window_size'],
                        paper_version=params["paper_version"])


    def initialise(self, latent_dim, data_dim, time_length,
                 encoder_sizes=(64, 64),
                 decoder_sizes=(64, 64),
                 beta=1.0, M=10, K=1, 
                 kernel="cauchy", sigma=1., 
                 length_scale=1.0, kernel_scales=1, window_size=3, paper_version=False):
        """ Proposed GP-VAE model with Gaussian Process prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            :param encoder_sizes: layer sizes for the encoder network
            :param decoder_sizes: layer sizes for the decoder network
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
            :param kernel: Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
            :param: paper_version: bool indicating whether to use the paper's verion of the decoder
                                        paper only uses a network for the mean and holds variance at 1 for all time series
        """
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.time_length = time_length

        self.encoder = JointEncoder(latent_dim, encoder_sizes, window_size=window_size)
        self.decoder = GaussianDecoder(data_dim, decoder_sizes, paper_version=paper_version)

        self.beta = beta
        self.K = K
        self.M = M
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        # KL components
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None


    def __call__(self, x, with_var=False):
        mean = self.decode(self.encode(x).mean()).mean()
        if with_var:
            variance = self.decode(self.encode(x).mean()).variance()
            return mean, variance
        return mean


    # Fetch variance
    def get_variance(self, x):
        return variance = self.decode(self.encode(x).mean()).variance()


    # Draw sample from imputed posterior
    def sample(self, x, num_samples=1):
        return tf.convert_to_tensor([self.decode(self.encode(x).mean()).sample() for _ in num_samples])


    def decode(self, z):
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        return self.decoder(tf.transpose(z, perm=perm))


    def encode(self, x):
        x = tf.identity(x)  # cast x as Tensor just in case
        return self.encoder(x)


    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = tf.random_normal(shape=(num_samples, self.latent_dim))
        return self.decode(noise)
    

    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)


    def _compute_loss(self, x, m_mask=None, return_parts=False):
        """
            Do not provide m_mask for standard time series imputation tasks!!!
        """
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
            m_mask = tf.cast(m_mask, tf.bool)

        pz = self._get_prior() # p(z)
        qz_x = self.encode(x) # q(z|x) ie. q(z)
        z = qz_x.sample() 
        px_z = self.decode(z) # p(x|z)

        nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D), implement -E[log p(x|z)]
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            nll = tf.where(m_mask, tf.zeros_like(nll), nll)
        nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)

        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            weights = -nll - kl  # shape=(M*K*BS)
            weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

            elbo = self.reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = tf.reduce_mean(elbo)  # scalar
        else: # Do not use importance sampling
            # if K==1, compute KL analytically
            kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1, implement ELBO = E[log p(x|z)] - KL(q || p)
            elbo = tf.reduce_mean(elbo)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            return -elbo, nll, kl
        else:
            return -elbo


    def compute_nll(self, x, y=None, m_mask=None):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_sample = self.encode(x).sample()
        x_hat_dist = self.decode(z_sample)
        nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            nll = tf.where(m_mask, nll, tf.zeros_like(nll))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(nll)


    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only during evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_mean = self.encode(x).mean()
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = tf.round(x_hat_mean)
        mse = tf.math.squared_difference(x_hat_mean, y)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # inverse mask, set zeros for observed values
        return tf.reduce_sum(mse)


    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dim
            self.prior = tfd.MultivariateNormalTriL(loc=tf.zeros([self.latent_dim, self.time_length]), scale_tril=tf.linalg.cholesky(kernel_matrix_tiled))
            # Below is depracated
            #self.prior = tfd.MultivariateNormalFullCovariance(
                #loc=tf.zeros([self.latent_dim, self.time_length]),
                #covariance_matrix=kernel_matrix_tiled)
        return self.prior
    

    def get_trainable_vars(self):
        self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
                          tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
        return self.trainable_variables


    def kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        squared_frobenius_norm = lambda x: tf.reduce_sum(tf.square(x), axis=[-2, -1])

        # Helper to identify if `LinearOperator` has only a diagonal component
        is_diagonal = lambda x: (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                                  isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                                  isinstance(x, tf.linalg.LinearOperatorDiag))

        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            if self.pz_scale_inv is None:
                self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
                self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv), self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

            if self.pz_scale_log_abs_determinant is None:
                self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

            a_shape = a.scale.shape
            if len(b.scale.shape) == 3:
                _b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
            else:
                _b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

            b_inv_a = _b_scale_inv @ a.scale.to_dense()

        # approx. 10x times faster on CPU than on GPU according to paper
        with tf.device('/cpu:0'):
            kl_div = (self.pz_scale_log_abs_determinant - 
                      a.scale.log_abs_determinant() +
                      0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
                              squared_frobenius_norm(b_inv_a) + 
                              squared_frobenius_norm(b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
        return kl_div

    # Only needed when using importance sampling
    def reduce_logmeanexp(self, x, axis, epsilon=1e-5):
        """Implementation of log-mean-exponent.
        Args:
            x: The tensor to reduce.
            axis: The dimensions to reduce.
            eps: Floating point scalar to avoid log-underflow -> found issue occassionally when using random (unseeded) init
        Returns:
            log_mean_exp: A tensor representing log(Avg{exp(x): x}).
        """
        x_max = tf.reduce_max(x, axis=axis, keepdims=True)
        return tf.math.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis, keepdims=True) + epsilon) + x_max

