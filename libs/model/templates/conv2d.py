import numpy as np

from libs.model.nodetemplate import NodeTemplate
from libs.model_helpers import activators, linalg

class Conv2D(NodeTemplate):
    def __init__(self, F, K=1, P=0, flatten_output=False, **kwargs):
        self.layer_name = 'conv2d'
        self.P = P
        self.F = F
        self.K = K
        self.flatten_output = flatten_output
        super().__init__(self.layer_name, activators.relu, activators.drelu, **kwargs)

    def f(self, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        z = self.z(x, kernel, b)

        return activators.relu(z)

    def z(self, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        z = np.zeros(self.z_shape(x.shape))

        xPadded = linalg.pad_to_axis(x, self.P, axis=1)
        xPadded = linalg.assert_shape(xPadded)

        kernel_count = kernel.shape[-1]
        for i in range(kernel_count):
            filter = kernel[:, :, :, i]
            result = linalg.convolve(xPadded, filter)
            result = result + b[i]
            z[:, :, i] = linalg.reduce_shape(result)

        if self.flatten_output is True:
            z = z.flatten()

        return z

    def from_upstream(self, upstream: np.ndarray, x: np.ndarray, kernel: np.ndarray, b: np.ndarray):
        P = self.P

        # Pad along second axis
        xPadded = linalg.pad_to_axis(x, P, axis=1)
        xPadded = linalg.assert_shape(xPadded) # To 3D

        z = self.z(x, kernel, b) # conv(x) + b
        z = z.flatten() if self.flatten_output is True else z

        da_dz = activators.drelu(z)
        dc_da_dz = upstream * da_dz

        dc_da_dz = dc_da_dz.reshape(self.z_shape(x.shape))  # Upstream is flattened so it should be reshaped assuming input is square

        dc_daL = np.zeros(x.shape)
        dc_dw = np.zeros(kernel.shape)
        depth = kernel.shape[2]
        for i in range(self.K):
            dc_da_dz_slice = linalg.assert_shape(dc_da_dz[:, :, i])
            result = linalg.convolve(xPadded, dc_da_dz_slice)
            dc_dw[:, :, :, i] = result

            filter = kernel[:, :, :, i]
            flipped_filter = np.flip(filter)
            result = linalg.full_convolve(dc_da_dz_slice, flipped_filter)
            result = result[P:-P, P:-P]  # Remove padding
            # result = result[:, :, 0]
            result = linalg.match_shape(result, dc_daL)
            dc_daL += result

        # dc_da_dz = dc_da_dz.flatten() if self.flatten_output is True else dc_da_dz  # Correct shape
        dc_da_dz = np.einsum('ijk -> k', dc_da_dz)
        return dc_dw, dc_da_dz, dc_daL


    def f_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape
        shape = self.z_shape(input_shape)

        return shape if self.flatten_output is False else np.prod(shape)

    def w_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        flat_shape = (self.F, self.F)
        depth_shape = flat_shape + (1,) if len(input_shape) <= 2 else flat_shape + (input_shape[-1],)
        return depth_shape + (self.K,)

    def z_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        N = input_shape[0] + 2 * self.P - self.F + 1
        shape = (N, N) + (self.K,)

        return shape

    def b_shape(self, input_shape=None) -> tuple:
        input_shape = input_shape or self.input_shape

        return (self.K,)
