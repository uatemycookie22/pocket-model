import tensorflow as tf
import numpy as np


def load_mnist():
    return tf.keras.datasets.mnist.load_data()


def load_dummy0():
    x_train = np.random.random((60000, 1))
    y_train = np.array([sample_x*2 for sample_x in x_train])

    return (x_train, y_train), (None, None)
