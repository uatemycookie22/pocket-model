import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

