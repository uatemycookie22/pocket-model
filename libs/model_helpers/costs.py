import numpy as np


def abs_squared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.square(x-y)


def dabs_squared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2*(x-y)
