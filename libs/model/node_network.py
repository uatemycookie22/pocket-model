import math

import numpy as np
from libs.model.nodetemplate import NodeTemplate


class NodeNetwork:
    def __init__(self):
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.layer_templates: list[NodeTemplate] = []
        self.f_shapes: list[tuple] = []

    def __str__(self):
        return self.weights

    def from_layer(self, layer: NodeTemplate):
        w_shape = None
        f_shape = None

        if len(self.layer_templates) > 0:
            w_shape = layer.w_shape(self.f_shapes[-1])
            f_shape = layer.f_shape(self.f_shapes[-1])
        else:
            w_shape = layer.w_shape()
            f_shape = layer.f_shape()

        self.f_shapes.append(f_shape)

        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)
        weights = np.random.uniform(-limit, limit, size=w_shape)
        biases = np.random.uniform(-limit, limit, size=f_shape)

        self.append_layer(
            layer,
            weights,
            biases
        )

    def append_layer(self, layer: NodeTemplate, w: np.ndarray, b: np.ndarray):
        self.weights.append(w)
        self.biases.append(b)
        self.layer_templates.append(layer)

    # Given input x, feed the network
    def feed_forward(self, x: np.ndarray):
        z = x
        for w, l, b in zip(self.weights, self.layer_templates, self.biases):
            z = l.f(z, w, b)

        return z

    # Same as feed_forwards but store the activations

    def feed_forwards(self, a: np.ndarray):
        activations = [a]
        for b, w, activator in zip(self.biases, self.weights, self.layer_templates):
            a = activator.f(a, w, b)
            activations.append(a)
        return activations
