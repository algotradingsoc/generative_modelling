import tensorflow as tf
import numpy as np


def make_nn(hidden_sizes, output_size, activation=tf.keras.activations.relu):
    NNLayers = [tf.keras.layers.Dense(h, activation=activation, dtype=tf.float32) for h in hidden_sizes]
    NNLayers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(NNLayers)


class Decoder(tf.keras.Model):
    def __init__(self, hidden_sizes, output_size, activation=tf.keras.activations.relu):
        super(Decoder, self).__init__()
        self.network = make_nn(hidden_sizes, output_size, activation=activation)

    def call(self, z):
        out = self.network(z)
        return out


class Encoder(tf.keras.Model):
    def __init__(self, hidden_sizes, output_size, activation=tf.keras.activations.relu):
        super(Encoder, self).__init__()
        self.network = make_nn(hidden_sizes, output_size, activation=tf.keras.activations.relu)

    def call(self, inputs, training=False):
        return self.network(inputs)


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_dim, encoder_layers, decoder_layers, activation=tf.keras.activations.relu):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder([input_size]+encoder_layers, self.latent_dim*2, activation=activation)
        self.decoder = Decoder([self.latent_dim]+decoder_layers, input_size, activation=activation)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(0.5*logvar) + mean

    def call(self, x):
        mean, logvar = tf.split(self.encoder(x), 2, axis=-1)
        z = self.reparameterize(mean, logvar)
        output = self.decoder(z)
        return output

    def sample(self, eps=None, batch_size=100):
        if eps is None:
            eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return self.decoder(eps)

class FlattenImage(tf.keras.Model):
    def __init__(self, shape, model, model_kwargs):
        super(FlattenImage, self).__init__()
        self.shape = shape
        self.flat_size = 1
        for i in self.shape:
            self.flat_size *= i
        self.vae = model(input_size=self.flat_size, **model_kwargs)

    def call(self, inputs, training=False):
        x = tf.keras.layers.Flatten()(inputs)
        x = self.vae(x)
        x = tf.reshape(x, (-1, *self.shape, 1))
        return x


