import numpy as np
from libs.model.layertemplate import LayerTemplate


class Network:
    def __init__(self):
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.activators = []
        self.dactivators = []
        self.activations = []

    def __str__(self):
        return self.weights

    def appendr_layer(self, layer: LayerTemplate):
        self.append_layer(
            layer,
            np.random.random((layer.k, layer.n)),
            np.random.random((layer.k,))
        )

    def append_layer(self, layer: LayerTemplate, w: np.ndarray, b: np.ndarray):
        assert w.shape == (layer.k, layer.n), "Weights data must be same shape as template"
        assert b.shape == (layer.k,), "Biases data must be same shape as template"

        if len(self.weights) > 0:
            prev_weights_shape = self.weights[len(self.weights)-1].shape
            prev_biases_shape = self.biases[len(self.weights)-1].shape

            assert layer.n == prev_weights_shape[0],\
                "n must be the same number of weight rows as previous layer"

            assert layer.n == prev_biases_shape[0],\
                "n must be the same number of biases as previous layer"

        self.weights.append(w)
        self.biases.append(b)
        self.activators.append(layer.activator)
        self.dactivators.append(layer.dactivator)

    # Given vector a, 'feed' the network
    def feed_forward(self, a: np.ndarray):
        for b, w, activator in zip(self.biases, self.weights, self.activators):
            a = activator(np.dot(w, a) + b)
        return a

    # TODO: Can be optimized into list comprehension
    def feed_forwards(self, a: np.ndarray):
        activations = [a]
        for b, w, activator in zip(self.biases, self.weights, self.activators):
            a = activator(np.dot(w, a) + b)
            activations.append(a)
        return activations



