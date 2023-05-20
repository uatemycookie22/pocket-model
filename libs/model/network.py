import numpy as np
from libs.model.layertemplate import LayerTemplate


class Network:
    def __init__(self):
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.activators = []
        self.dactivators = []
        self.activations = []
        self.layer_templates: list[LayerTemplate] = []

    def __str__(self):
        return self.weights

    def appendr_layer(self, layer: LayerTemplate):
        prev_Ln: int = layer.prev_n if len(self.weights) == 0 else self.weights[-1].shape[0]

        assert prev_Ln is not None and prev_Ln > 0

        self.append_layer(
            layer,
            np.random.uniform(-1, 1, (layer.n, prev_Ln)),
            np.random.uniform(-1, 1, (layer.n,))
        )

    def append_layer(self, layer: LayerTemplate, w: np.ndarray, b: np.ndarray):
        assert w.shape == (layer.n, w.shape[1]), "Weights data must be same shape as template"
        assert b.shape == (layer.n,), "Biases data must be same shape as template"

        if len(self.weights) > 0:
            prev_weights_shape = self.weights[len(self.weights)-1].shape
            prev_biases_shape = self.biases[len(self.weights)-1].shape

            assert w.shape[1] == prev_weights_shape[0],\
                f"n must be the same number of weight rows as previous layer n: {w.shape} prev n: {prev_weights_shape}"

            assert w.shape[1] == prev_biases_shape[0],\
                f"n must be the same number of biases as previous layer n: {w.shape} prev n: {prev_biases_shape}"

        self.weights.append(w)
        self.biases.append(b)
        self.activators.append(layer.activator)
        self.dactivators.append(layer.dactivator)
        self.layer_templates.append(layer)

    # Given vector a, 'feed' the network
    def feed_forward(self, a: np.ndarray):
        for b, w, activator in zip(self.biases, self.weights, self.activators):
            a = activator(np.dot(w, a) + b)
        return a

    def feed_forwards(self, a: np.ndarray):
        activations = [a]
        for b, w, activator in zip(self.biases, self.weights, self.activators):
            a = activator(np.dot(w, a) + b)
            activations.append(a)
        return activations



