import tensorflow as tf

"""
    Some kernels used in GP-VAE
"""

def rbf_kernel(T, length_scale=1.0):
    X = tf.range(T, dtype=tf.float32)
    X_in, X_out = tf.expand_dims(X, 0), tf.expand_dims(X, 1)
    distance_matrix = tf.math.squared_difference(X_in, X_out)
    scaled_distance_matrix = distance_matrix / length_scale ** 2
    K = tf.math.exp(-scaled_distance_matrix)
    return K

def cauchy_kernel(T, sigma=1.0, length_scale=1.0):
    X = tf.range(T, dtype=tf.float32)
    X_in, X_out = tf.expand_dims(X, 0), tf.expand_dims(X, 1)
    distance_matrix = tf.math.squared_difference(X_in, X_out)
    scaled_distance_matrix = distance_matrix / length_scale ** 2
    K = tf.math.divide(sigma, (scaled_distance_matrix + 1.))

    alpha = 0.001
    I = tf.eye(num_rows=K.shape.as_list()[-1])
    return K + alpha * I

def matern_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / tf.cast(tf.math.sqrt(length_scale), dtype=tf.float32)
    K = tf.math.exp(-distance_matrix_scaled)
    return K


def linear_kernel(T, sigma=0.0):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    K = tf.multiply(xs_in, xs_out) + sigma**2
    return K