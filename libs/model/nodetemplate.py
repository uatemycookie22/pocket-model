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

    def b_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        return self.f_shape(input_shape)


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


class Sigmoid(NodeTemplate):
    def __init__(self, current_n, **kwargs):
        self.layer_name = 'sigmoid'
        self.current_n = current_n
        self.shape = DenseShape(current_n)

        super().__init__(self.layer_name, activators.sigmoid, activators.dsigmoid, **kwargs)

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


class Linear(NodeTemplate):
    def __init__(self, current_n, c=1, **kwargs):
        self.layer_name = 'linear'
        self.current_n = current_n
        self.shape = DenseShape(current_n)
        self.c = c

        super().__init__(
            self.layer_name,
            lambda x: activators.linear(x, c),
            lambda x: activators.dlinear(x, c),
            **kwargs
        )

    def f(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b
        return self.fun(z)

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        z = linalg.matvec(w, x) + b

        da_dz = self.fun_grad(z)
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
