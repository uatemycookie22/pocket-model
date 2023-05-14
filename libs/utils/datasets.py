import tensorflow as tf


def load_mnist():
    return tf.keras.datasets.mnist.load_data()

