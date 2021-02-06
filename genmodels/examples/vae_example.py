import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from genmodels.vae.vae import *

def get_mnist(dataset=tf.keras.datasets.fashion_mnist.load_data, val_split=None, normalise_factor=255.0):
    (x_train, y_train), (x_test, y_test) = dataset()
    x_train = x_train[..., np.newaxis] / normalise_factor
    x_test = x_test[..., np.newaxis] / normalise_factor
    if val_split is not None:
        val_length = int(len(x_train) * val_split)
        x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
        x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]
        x_val = np.float32(x_val)
    else:
        x_val = y_val = None

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    train = (x_train, y_train)
    val = (x_val, y_val)
    test = (x_test, y_test)
    return train, val, test

(x_train, _), (x_val, _), (x_test, _) = get_mnist(val_split=0.2)

model = FlattenImage((28,28), VAE, {'latent_dim':28, 'encoder_layers':[128], 'decoder_layers':[128]})

x = model(x_train)
model.compile(optimizer='sgd', loss=tf.keras.losses.KLDivergence())
h = model.fit(x_train, x_train, validation_data=(x_val, x_val), verbose=1, epochs=10)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])

plt.show()
