import unittest
from libs.model.network import Network
from fixtures import layers
from libs.model_helpers import activators
import numpy as np
from fixtures import inputs


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


if __name__ == '__main__':
    unittest.main()
