import numpy as np

from libs.model.nodetemplate import NodeTemplate
from libs.model_helpers import activators, linalg


class MaxPool2D(NodeTemplate):
    def __init__(self, F, S, K=1, P=0, flatten_output=False, **kwargs):
        self.layer_name = 'maxpool2d'
        self.P = P
        self.F = F
        self.K = K
        self.S = S
        self.pos = None
        self.flatten_output = flatten_output
        super().__init__(self.layer_name, activators.relu, activators.drelu, **kwargs)

    def f(self, x: np.ndarray, kernel: np.ndarray = None, b: np.ndarray = None):
        val, pos = linalg.maxpool(x, self.F, self.S)
        self.pos = pos

        if self.flatten_output is True:
            val = val.flatten()

        return val

    def z(self, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        pass

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, kernel = None, b = None):
        P = self.P

        val, pos = linalg.maxpool(x, self.F, self.S)

        upstream = upstream.reshape(val.shape)
        upstream = linalg.assert_shape(upstream)
        dc_daL = linalg.unpooling(upstream, pos, x.shape, self.S)

        # dc_daL = np.clip(dc_daL, 0, 1)
        # dc_da_dz = dc_da_dz.reshape(self.z_shape(x.shape))  # Upstream is flattened so it should be reshaped assuming input is square

        dc_da_dz = np.zeros(self.b_shape())
        dc_dw = np.zeros(self.w_shape())

        return dc_dw, dc_da_dz, dc_daL

    def f_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        shape = self.z_shape(input_shape)

        return shape if self.flatten_output is False else np.prod(shape)

    def w_shape(self, input_shape=None) -> tuple:
        return (1,)

    def z_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        N = (input_shape[0] + 2 * self.P - self.F) / self.S + 1

        assert N.is_integer(), "Output shape should be integer (W - F)/S + 1"

        N = int(N)

        shape = (N, N, 1) if len(input_shape) < 3 else (N, N) + (input_shape[-1],)

        return shape

    def b_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        return (self.K,)

    def rand_w(self, input_shape=None):
        input_shape = input_shape or self.input_shape
        w_shape = self.w_shape(input_shape)

        weights = np.random.randn(*w_shape) / np.sqrt(np.prod(input_shape))
        return weights
