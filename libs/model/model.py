import numpy as np

from libs.model.layertemplate import LayerTemplate
from libs.model.network import Network
from libs.model_helpers import linalg


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
        return np.sum(cost_func(a, label))

    def backprop(self, input: np.ndarray, label: np.ndarray, dcost_func):
        activations = self.nn.feed_forwards(input)

        dc_dw = []
        dc_db = []
        dc_da = []

        dc_daL = None
        for L in range(len(self.nn.weights) - 1, -1, -1):
            w = self.nn.weights[L]  # w(L): j X k
            b = self.nn.biases[L]  # b(L): j X 1
            z = np.dot(w, activations[L]) + b  # z(L): j X 1
            a = activations[L+1]  # a(L): j X 1

            # Find derivative of cost with respect to its weights
            if dc_daL is None:
                dc_daL = dcost_func(a, label) # j X 1

            dc_da_dz = linalg.dcost_db(dc_daL, self.nn.dactivators[L](z))
            dc_db.append(dc_da_dz)  # j X 1
            dc_dw.append(linalg.dcost_dw(dc_da_dz, activations[L]))  # j X k
            dc_daL = linalg.dcost_dpreva(dc_da_dz, w)  # j X 1
            dc_da.append(dc_daL)  # j X 1

        return dc_dw, dc_db, dc_da
