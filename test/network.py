import unittest

import libs.model.templates.conv2d
from libs.model.network import Network
from fixtures import layers
from libs.model_helpers import activators
from libs.model import nodetemplate
import numpy as np
from fixtures import inputs
from libs.model_helpers import linalg
from libs.model.node_network import NodeNetwork

class NetworkTest(unittest.TestCase):
    def test_append_randomized_layer(self):
        sut = Network()

        sut.appendr_layer(layers.relu_4_9)

        self.assertEqual(1, len(sut.weights))
        self.assertEqual((4, 9), sut.weights[0].shape)
        self.assertEqual((4,), sut.biases[0].shape)
        self.assertEqual(activators.relu, sut.activators[0])

    def test_feedforward_input_validation(self):
        sut = Network()

        sut.appendr_layer(layers.relu_4_9)
        sut.appendr_layer(layers.relu_4_4)
        sut.appendr_layer(layers.relu_1_4)

        self.assertRaises(ValueError, sut.feed_forward, inputs.fiven)
        self.assertRaises(ValueError, sut.feed_forward, inputs.onen)

        sut.feed_forward(inputs.ninen)
        sut.feed_forward(inputs.niner)

    def test_append_input_validation(self):
        sut = Network()
        sut.append_layer(layers.relu_1_9, np.zeros((1, 9)), np.zeros((1,)))

        sut = Network()
        self.assertRaises(
            AssertionError,
            sut.append_layer,
            layers.relu_1_9, np.zeros((1, 1)), np.zeros((9,))
        )

        sut = Network()
        self.assertRaises(
            AssertionError,
            sut.append_layer,
            layers.relu_1_9, np.zeros((1, 9)), np.zeros((9,))
        )

        sut = Network()
        sut.append_layer(layers.relu_4_9, np.zeros((4, 9)), np.zeros((4,)))
        sut.append_layer(layers.relu_4_4, np.zeros((4, 4)), np.zeros((4,)))
        sut.append_layer(layers.relu_4_4, np.zeros((4, 4)), np.zeros((4,)))
        sut.append_layer(layers.relu_1_4, np.zeros((1, 4)), np.zeros((1,)))

        sut = Network()
        sut.append_layer(layers.relu_4_9, np.zeros((4, 9)), np.zeros((4,)))
        self.assertRaises(
            AssertionError,
            sut.append_layer,
            layers.relu_4_9, np.zeros((4, 9)), np.zeros((4,))
        )
        self.assertRaises(
            AssertionError,
            sut.append_layer,
            layers.relu_1_9, np.zeros((1, 9)), np.zeros((1,))
        )
        sut.append_layer(layers.relu_1_4, np.zeros((1, 4)), np.zeros((1,)))

    def test_feedforward_simple(self):
        sut = Network()

        # Test weights
        zero_bias = np.zeros(1, )
        sut.append_layer(layers.relu_1_1, np.zeros((1, 1)), zero_bias)

        expected = layers.relu_1_1.activator(np.dot(np.zeros((1, 1)), inputs.onen))
        actual = sut.feed_forward(inputs.onen)
        self.assertEqual(expected, actual)

        # Append layer with weight = 1 bias = 1
        plusone_bias = np.array([1])
        sut.append_layer(layers.relu_1_1, np.array([[1]]), plusone_bias)

        expected = layers.relu_1_1.activator(plusone_bias)
        actual = sut.feed_forward(inputs.onen)
        self.assertEqual(expected, actual)

        # Append layer with weight = 1 bias = 0 (should give same result as above assertion)
        sut.append_layer(layers.relu_1_1, np.array([[1]]), zero_bias)

        expected = layers.relu_1_1.activator(plusone_bias)
        actual = sut.feed_forward(inputs.onen)
        self.assertEqual(expected, actual)

    def test_feedforward_simple_randomized(self):
        sut = Network()

        # Randomized weight bias and input
        weightr = np.random.random((1,1))
        biasr = np.random.random((1,))
        inputr = np.random.random((1,))

        # Test weights
        zero_bias = np.zeros(1, )
        sut.append_layer(layers.relu_1_1, weightr, zero_bias)

        expected = layers.relu_1_1.activator(np.dot(weightr, inputr))
        actual = sut.feed_forward(inputr)
        self.assertEqual(expected, actual)

        # Test biases
        one_weight = np.array([[1]])
        sut.append_layer(layers.relu_1_1, one_weight, biasr)

        expected = layers.relu_1_1.activator(expected + biasr)
        actual = sut.feed_forward(inputr)
        self.assertEqual(expected, actual)

    def test_node_feedforwardconv(self):
        x = np.arange(25).reshape((5,5))
        N = x.shape[0] * x.shape[1]

        w1 = np.ones((3, 3))
        w2 = np.ones((2, N))
        w3 = np.ones((2, 2))

        xPadded = np.pad(x, [(1, 1), (1, 1)])

        z = linalg.convolve(xPadded, w1).flatten()
        z = w2.dot(z)
        z1 = w3.dot(z).sum()

        layer1 = libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True)
        layer2 = nodetemplate.ReLU(2)
        layer3 = nodetemplate.ReLU(2)

        z = layer1.f(x, kernel=w1)
        z = layer2.f(z, w=w2)
        z2 = layer3.f(z, w=w3).sum()

        self.assertEqual(z1, z2)

        W = [w1, w2, w3]
        L: list[nodetemplate.NodeTemplate] = [layer1, layer2, layer3]

        z = x
        for w, l in zip(W, L):
            z = l.f(z, w)
        z = z.sum()

        self.assertEqual(z, z2)

        nn = NodeNetwork()

        nn.append_layer(layer1, w1, np.zeros(25))
        nn.append_layer(layer2, w2, np.zeros(2))
        nn.append_layer(layer3, w3, np.zeros(2))

        z = nn.feed_forward(x).sum()

        self.assertEqual(z, z2)

        nn = NodeNetwork()
        for l in L:
            nn.from_layer(l)

        z = nn.feed_forward(x).sum()



if __name__ == '__main__':
    unittest.main()
