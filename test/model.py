import unittest
from libs.model.model import Model
from libs.model.network import Network
from fixtures import layers
from libs.model_helpers import activators
import numpy as np
from fixtures import inputs
from libs.model_helpers import costs
from libs.model import layertemplate
from libs.utils import datasets

class ModelTest(unittest.TestCase):
    def test_build(self):
        sut = Model()
        sut.build([
            layertemplate.ReLU(1, 3),
            layertemplate.ReLU(3),
            layertemplate.ReLU(5),
            layertemplate.ReLU(9),
        ])

    def test_build_raise(self):
        sut = Model()
        self.assertRaises(
            AssertionError,
            sut.build,
            [layertemplate.ReLU(1), layertemplate.ReLU(1)]
        )

    def test_sample_cost_simple(self):
        sut = Model()
        sut.build([
            layertemplate.ReLU(1, 1)
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
            layertemplate.ReLU(1, 3)
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
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([0])
        expected = ([np.array([2])], [np.array([2])], [np.array([2])])
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([0])
        expected = ([np.array([2])], [np.array([2])], [np.array([2])])
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

        self.assertEqual(expected, actual)

        y = np.array([2])
        expected = ([np.array([-2])], [np.array([-2])], [np.array([-2])])
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

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
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

        self.assertEqual(expected, actual)

        # 1 layer weight = 2
        nn = Network()

        zero_bias = np.zeros(1,)
        y = np.array([1/2])
        weight = np.array([[2]])

        nn.append_layer(layers.relu_1_1, weight, zero_bias)

        sut.nn = nn

        expected = ([np.array([3])], [np.array([3])], [np.array([6])])
        actual = sut.step(inputs.onen, y, costs.dabs_squared)

        self.assertEqual(expected, actual)

        # 3 to 1
        nn = Network()

        zero_bias = np.zeros(1, )
        y = np.array([1])

        nn.append_layer(layers.relu_1_3, np.array([[1, 2, 3]]), zero_bias)

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = \
            sut.step(inputs.threen, y, costs.dabs_squared)
        expected = [np.array([[26, 52, 78]])]

        self.assertTrue(np.array_equal(expected, actual_weights), f"w expected: {expected} actual: {actual_weights}")

        expected = [np.array([26])]
        self.assertTrue(np.array_equal(expected, actual_biases), "b")

        expected = [np.array([26, 52, 78])]
        self.assertTrue(np.array_equal(expected, actual_activations)
                        , f"a expected {expected} actual: {actual_activations}")

    def test_backprop_weights_multilayer(self):
        sut = Model()

        # 2 layer
        nn = Network()

        zero_bias = np.zeros(1, )
        y = np.array([1 / 2])

        nn.append_layer(layers.relu_1_1, np.array([[1]]), zero_bias)
        nn.append_layer(layers.relu_1_1, np.array([[2]]), zero_bias)

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = \
            sut.step(inputs.onen, y, costs.dabs_squared)

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_weights, "w")

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_biases, "b")

        expected = [np.array([6]), np.array([6])]
        self.assertEqual(expected, actual_activations, "a")

    def test_backprop_shape_2layer(self):
        sut = Model()

        # 2 layer
        nn = Network()

        zero_bias = np.zeros((4,))
        y = np.zeros((4,))

        nn.append_layer(
            layers.relu_4_4,
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]),
            zero_bias
        )

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = \
            sut.step(inputs.fourn, y, costs.dabs_squared)

        dexpected = [(4, 4)]
        for dweight, expected in zip(actual_weights, dexpected):
            self.assertEqual(expected, dweight.shape, f"shape expected: {expected}\nactual:{dweight.shape}")

        dexpected = [(4,)]
        for dbias, expected in zip(actual_biases, dexpected):
            self.assertEqual(expected, dbias.shape, f"shape expected: {expected}\nactual:{dbias.shape}")

        dexpected = [(4,)]
        for da, expected in zip(actual_activations, dexpected):
            self.assertEqual(expected, da.shape, f"shape expected: {expected}\nactual:{da.shape}")

    def test_backprop_shape_3layer(self):
        sut = Model()

        nn = Network()

        zero_bias = np.zeros((4,))
        y = np.zeros((4,))
        id_weight =  np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
        ])

        nn.append_layer(layers.relu_4_4, id_weight, zero_bias)
        nn.append_layer(layers.relu_4_4, id_weight, zero_bias)

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = \
            sut.step(inputs.fourn, y, costs.dabs_squared)

        dexpected = [(4, 4), (4, 4)]
        for dweight, expected in zip(actual_weights, dexpected):
            self.assertEqual(expected, dweight.shape, f"shape expected: {expected}\nactual:{dweight.shape}")

        dexpected = [(4,), (4,)]
        for dbias, expected in zip(actual_biases, dexpected):
            self.assertEqual(expected, dbias.shape, f"shape expected: {expected}\nactual:{dbias.shape}")

        dexpected = [(4,), (4,)]
        for da, expected in zip(actual_activations, dexpected):
            self.assertEqual(expected, da.shape, f"shape expected: {expected}\nactual:{da.shape}")

    def test_mnist_time(self):
        np.random.seed()
        (x_train, y_train), (_, _) = datasets.load_mnist()
        x: np.ndarray = x_train[0]
        y: np.ndarray = y_train[0]

        sut = Model()
        sut.build([
            layertemplate.ReLU(12, x.shape[0] * x.shape[1]),
            layertemplate.ReLU(12),
            layertemplate.Sigmoid(9),
        ])

        xprocessed = x.flatten()/np.amax(x)

        sut.sample(xprocessed, y, costs.abs_squared)
        sut.step(xprocessed, y, costs.dabs_squared)




if __name__ == '__main__':
    unittest.main()
