import numpy as np

from libs.model_helpers import activators
from libs.model_helpers import linalg


class NodeTemplate:
    def __init__(self, layer_name: str, fun, fun_grad, input_shape: tuple[int] = ()):
        self.layer_name = layer_name
        self.fun = fun
        self.fun_grad = fun_grad
        self.input_shape = input_shape

    def f_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        return input_shape

    def w_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        return input_shape


class ReLU(NodeTemplate):
    def __init__(self, current_n, **kwargs):
        self.layer_name = 'relu'
        self.current_n = current_n
        self.shape = DenseShape(current_n)

        super().__init__(self.layer_name, activators.relu, activators.drelu, **kwargs)

    def f(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b
        return activators.relu(z)

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b

        da_dz = activators.drelu(z)
        dc_da_dz = upstream * da_dz
        dc_dw = linalg.dcost_dw(dc_da_dz, x)
        dc_daL = linalg.dcost_dpreva(dc_da_dz, w)  # j X 1

        return dc_dw, dc_da_dz, dc_daL

    def f_shape(self, input_shape=None) -> tuple:
        return self.shape.f_shape()

    def w_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        return self.shape.w_shape(input_shape)


class Conv2D(NodeTemplate):
    def __init__(self, F, P=0, flatten_output=False, **kwargs):
        self.layer_name = 'conv2d'
        self.P = P
        self.F = F
        self.flatten_output = flatten_output
        super().__init__(self.layer_name, activators.relu, activators.drelu, **kwargs)

    def f(self, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        xPadded = np.pad(x, [(self.P, self.P), (self.P, self.P)])
        z = linalg.convolve(xPadded, kernel)

        if self.flatten_output is True:
            z = z.flatten()

        return activators.relu(z + b)

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        xPadded = np.pad(x, [(self.P, self.P), (self.P, self.P)])
        z = linalg.convolve(xPadded, kernel)
        z = z.flatten() if self.flatten_output is True else z
        z += b

        da_dz = activators.drelu(z)
        dc_da_dz = upstream * da_dz

        N = x.shape[0] + 2 * self.P - self.F + 1
        dc_da_dz = dc_da_dz.reshape((N, N))  # Upstream is flattened so it should be reshaped assuming input is square
        dc_dw = linalg.convolve(xPadded, dc_da_dz)

        flipped_kernel = np.flip(kernel)
        dc_daL = linalg.full_convolve(flipped_kernel, dc_da_dz)

        return dc_dw, dc_da_dz.flatten() if self.flatten_output is True else dc_da_dz, dc_daL

    def f_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        N = input_shape[0] + 2 * self.P - self.F + 1
        shape = (N, N)

        return shape if self.flatten_output is False else (N * N)

    def w_shape(self, input_shape=None) -> tuple:
        return self.F, self.F


class Sigmoid(NodeTemplate):
    def __init__(self, current_n):
        self.layer_name = 'sigmoid'
        self.current_n = current_n
        self.shape = DenseShape(current_n)

        super().__init__(self.layer_name, activators.sigmoid, activators.dsigmoid)

    def f(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b
        return activators.sigmoid(z)

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b

        da_dz = activators.dsigmoid(z)
        dc_da_dz = upstream * da_dz
        dc_dw = linalg.dcost_dw(dc_da_dz, x)
        dc_daL = linalg.dcost_dpreva(dc_da_dz, w)  # j X 1

        return dc_dw, dc_da_dz, dc_daL

    def f_shape(self, input_shape=None) -> tuple:
        return self.shape.f_shape()

    def w_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        return self.shape.w_shape(input_shape)



class DenseShape:
    def __init__(self, current_n):
        self.current_n = current_n

    def f_shape(self) -> tuple:
        return self.current_n

    def w_shape(self, input_shape) -> tuple:
        input_shape = (self.current_n, input_shape)

        return input_shape
