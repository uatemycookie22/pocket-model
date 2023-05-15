import unittest
from libs.model.model import Model
from libs.model.network import Network
from fixtures import layers
from libs.model_helpers import activators
import numpy as np
from fixtures import inputs
from libs.model_helpers import costs


class ModelTest(unittest.TestCase):
    def test_build(self):
        sut = Model()
        sut.build([
            layers.relu_1_3,
            layers.relu_3_1,
            layers.relu_1_3,
        ])

    def test_sample_cost_simple(self):
        sut = Model()
        sut.build([
            layers.relu_1_1
        ])

        y = np.array([1])

        actual = sut.sample(inputs.onen, y, costs.abs_squared)
        expected = costs.abs_squared(activators.relu(sut.nn.weights[0] + sut.nn.biases[0]), y)

        self.assertEqual(expected, actual)

        actual = sut.sample(inputs.one_neg0, y, costs.abs_squared)
        expected = costs.abs_squared(activators.relu(-sut.nn.weights[0] + sut.nn.biases[0]), y)

        self.assertEqual(expected, actual)

        actual = sut.sample(inputs.one_0, y, costs.abs_squared)
        expected = costs.abs_squared(activators.relu(4 * sut.nn.weights[0] + sut.nn.biases[0]), y)

        self.assertEqual(expected, actual)

    def test_sample_cost(self):
        sut = Model()
        sut.build([
            layers.relu_1_3
        ])

        y = np.array([1])

        actual = sut.sample(inputs.threen, y, costs.abs_squared)
        expected = costs.abs_squared(
            activators.relu(
                np.dot(sut.nn.weights, inputs.threen) + sut.nn.biases[0]
            ),
            y
        )

        self.assertEqual(expected, actual)

    def test_backprop_label(self):
        sut = Model()

        nn = Network()

        zero_bias = np.zeros(1,)
        id_weight = np.array([[1]])
        nn.append_layer(layers.relu_1_1, id_weight, zero_bias)

        sut.nn = nn

        y = np.array([1])
        expected = ([np.array([0])], [np.array([0])], [np.array([0])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([0])
        expected = ([np.array([2])], [np.array([2])], [np.array([2])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([0])
        expected = ([np.array([2])], [np.array([2])], [np.array([2])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([2])
        expected = ([np.array([-2])], [np.array([-2])], [np.array([-2])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

    def test_backprop_weights(self):
        sut = Model()

        nn = Network()

        zero_bias = np.zeros(1,)
        y = np.array([1/2])
        id_weight = np.array([[1]])

        nn.append_layer(layers.relu_1_1, id_weight, zero_bias)

        sut.nn = nn

        expected = ([np.array([1])], [np.array([1])], [np.array([1])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

        # 1 layer weight = 2
        nn = Network()

        zero_bias = np.zeros(1,)
        y = np.array([1/2])
        weight = np.array([[2]])

        nn.append_layer(layers.relu_1_1, weight, zero_bias)

        sut.nn = nn

        expected = ([np.array([3])], [np.array([3])], [np.array([6])])
        actual = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        self.assertEqual(expected, actual)

        # 2 layer
        nn = Network()

        zero_bias = np.zeros(1,)
        y = np.array([1 / 2])

        nn.append_layer(layers.relu_1_1, np.array([[1]]), zero_bias)
        nn.append_layer(layers.relu_1_1, np.array([[2]]), zero_bias)

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = sut.backprop(inputs.onen, y, costs.abs_squared, costs.dabs_squared)

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_weights, "w")

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_weights, "b")

        expected = [np.array([6]), np.array([6])]
        self.assertEqual(expected, actual_weights, "a")


if __name__ == '__main__':
    unittest.main()
