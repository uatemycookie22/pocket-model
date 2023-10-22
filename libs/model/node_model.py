import time
from typing import Any

import numpy as np
from libs.model.nodetemplate import NodeTemplate
from libs.model import layertemplate
from libs.model.network import Network
from libs.model.node_network import NodeNetwork
from libs.model_helpers import linalg
from libs.model_helpers import costs
from libs.plotters.model_plots import CostRT, Eval
from libs.utils import io
from datetime import datetime


class NodeModel:
    def __init__(self, costf = costs.abs_squared, dcostf = costs.dabs_squared):
        self.nn = NodeNetwork()
        self.costf = costf
        self.dcostf = dcostf
        self._built = False

    # Given a list of layers (n nodes for layer L and activator),
    # build randomized neural network
    def build(self, layers: list[NodeTemplate]):
        assert self._built is not True, "_built must be False"
        for layer in layers:
            self.nn.from_layer(layer)

        self._built = True

    def predict(self, x):
        return self.nn.feed_forward(x)

    def sample(self, input: np.ndarray, label: np.ndarray, cost_func):
        a = self.nn.feed_forward(input)
        return np.sum(cost_func(a, label))

    def step(self, input: np.ndarray, label: np.ndarray, dcost_func):
        activations = self.nn.feed_forwards(input)

        dc_dw = []
        dc_db = []
        dc_da = []

        dc_daL = None
        for L in range(len(self.nn.weights) - 1, -1, -1):
            layer: Any = self.nn.layer_templates[L]
            w = self.nn.weights[L]
            b = self.nn.biases[L]

            # Find derivative of cost with respect to its weights
            if dc_daL is None:
                dc_daL = dcost_func(activations[L+1], label)

            dc_dwL, dc_dbL, dc_daL = layer.from_upstream(dc_daL, activations[L], w, b)

            dc_dw.append(dc_dwL)
            dc_da.append(dc_daL)
            dc_db.append(dc_dbL)

        return dc_dw, dc_db, dc_da