import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def drelu(x: np.ndarray) -> np.ndarray:
    return np.heaviside(x, 1)


def leaky_relu(x: np.ndarray, alpha=0) -> np.ndarray:
    return np.maximum(alpha*x, x)


def dleaky_relu(x: np.ndarray, alpha=0) -> np.ndarray:
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig*(1-sig)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.square(tanh(x))


def linear(x: np.ndarray, c=1) -> np.ndarray:
    return x * c


def dlinear(x: np.ndarray, c=1) -> np.ndarray:
    return np.full(x.shape, c)


def normal(x: np.ndarray) -> np.ndarray:
    return (x - x.mean())/x.std()


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
