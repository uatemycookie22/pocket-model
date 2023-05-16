import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def drelu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.minimum(1, x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig*(1-sig)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.square(tanh(x))

