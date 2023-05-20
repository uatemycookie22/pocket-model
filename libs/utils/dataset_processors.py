import numpy as np


def grayscale(x: np.ndarray) -> np.ndarray:
    return np.array([sample / np.amax(sample) for sample in x])


def flatten_2d(x: np.ndarray,  order='C') -> np.ndarray:
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]), order=order)


def norm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean())/x.std()


def one_hot_encode(data, classes):
    # Create an empty matrix with zero values
    one_hot_array = np.zeros((len(data), classes))

    # Place a 1 at the index corresponding to the class number
    one_hot_array[np.arange(len(data)), data] = 1

    return one_hot_array
