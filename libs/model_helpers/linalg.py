import numpy as np
import scipy


def dcost_dpreva(dc_db: np.ndarray, w: np.ndarray):
    return np.asarray([np.dot(np.array(w.T[k, :]), dc_db) for k in range(w.shape[1])])


def dcost_dw(dc_da_dz: np.ndarray, preva: np.ndarray):
    return np.outer(dc_da_dz, preva)


def dcost_db(dc_da: np.ndarray, da_dz: np.ndarray):
    return dc_da * da_dz


def avg_gradient(gradients: list[list[np.ndarray]]):
    avg_gradients = []
    gradients_size = len(gradients)
    gradient_size = len(gradients[0])

    for L in range(gradient_size):
        avg_gradient = gradients[0][L]

        for gradient in gradients[min(1, gradient_size):]:
            avg_gradient += gradient[L]

        avg_gradient = avg_gradient / gradients_size

        avg_gradients.append(avg_gradient)

    return avg_gradients


def convolve(m: np.ndarray, kernel: np.ndarray):
    return scipy.signal.correlate(m, kernel, 'valid')


def full_convolve(m: np.ndarray, kernel: np.ndarray):
    return scipy.signal.correlate(m, kernel, 'full')


def dconvolve(m: np.ndarray, dc_da: np.ndarray):
    return scipy.signal.correlate(m, dc_da, 'valid')


def matvec(m: np.ndarray, v: np.ndarray):
    return m.dot(v)


def pad_to_axis(x: np.ndarray, P: int, axis=0) -> np.ndarray:
    pad_width = ((P, P),) * (axis + 1)
    D = len(x.shape)
    empty_pad = (((0, 0),) * (D - axis))[:D]
    pad_width = pad_width + empty_pad
    return np.pad(x, pad_width[:D])


def assert_shape(x: np.ndarray) -> np.ndarray:
    if len(x.shape) <= 2:
        return np.expand_dims(x, axis=2)
    return x


def reduce_shape(x: np.ndarray) -> np.ndarray:
    if len(x.shape) > 2:
        return np.squeeze(x, axis=2)
    return x


# Matches x to y and returns x
def match_shape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x.shape) > len(y.shape):
        return reduce_shape(x)
    if len(x.shape) < len(y.shape):
        return assert_shape(x)
    return x
