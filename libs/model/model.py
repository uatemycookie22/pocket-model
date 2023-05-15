import numpy as np

from libs.model.layertemplate import LayerTemplate
from libs.model.network import Network


class Model:
    def __init__(self):
        self.nn = Network()
        self._built = False

    # Given a list of layers (n nodes for layer L and activator),
    # build randomized neural network
    def build(self, layers: list[LayerTemplate]):
        assert self._built is not True, "_built must be False"
        for layer in layers:
            self.nn.appendr_layer(layer)
        self._built = True

    def sample(self, input: np.ndarray, label: np.ndarray, cost_func):
        a = self.nn.feed_forward(input)
        return cost_func(a, label)

    def backprop(self, input: np.ndarray, label: np.ndarray, cost_func, dcost_func):
        print(input, label)

        activations = self.nn.feed_forwards(input)
        print(activations)

        dc_dw = []
        dc_db = []
        dc_da = []

        for L in range(len(self.nn.weights) - 1, -1, -1):
            print(L)

            # Find derivative of cost with respect to its weights
            w = self.nn.weights[L]  # w(L)
            b = self.nn.biases[L]  # b(L)
            z = np.dot(w, activations[L]) + b  # z(L)
            a = self.nn.activators[L](z)  # a(L)

            dc_da_dz = self.nn.dactivators[L](z) * dcost_func(a, label)
            dc_db.append(dc_da_dz)
            dc_dw.append(activations[L] * dc_da_dz)
            dc_da = w * dc_da_dz

        return dc_dw, dc_db, dc_da
