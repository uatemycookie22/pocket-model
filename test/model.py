import unittest

import libs.model.templates.conv2d
import libs.model.templates.maxpool2d
from libs.model.model import Model
from libs.model.network import Network
from libs.model.node_model import NodeModel
from libs.model.node_network import NodeNetwork
from libs.model import nodetemplate
from fixtures import layers
from libs.model_helpers import activators
import numpy as np
from fixtures import inputs
from libs.model_helpers import costs
from libs.model import layertemplate
from libs.utils import datasets
from utils import numeric_grad, cmp_arr


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

    def test_build_node(self):
        x = np.arange(25).reshape((5, 5))

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True),
            nodetemplate.ReLU(2),
            nodetemplate.ReLU(2),
            nodetemplate.Sigmoid(2),
        ])

        sut.predict(x)

    def test_build_node_multiconv(self):
        x = np.arange(25).reshape((5, 5))

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=5, P=2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.ReLU(2),
            nodetemplate.ReLU(2),
        ])

        sut.predict(x)




    def test_backprop_node(self):
        sut = NodeModel()

        nn = NodeNetwork()

        zero_bias = np.zeros(1, )
        id_weight = np.array([[1]])
        nn.append_layer(nodetemplate.ReLU(1), id_weight, zero_bias)

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

    def test_backprop_weights_multilayer_node(self):
        sut = NodeModel()

        # 2 layer
        nn = NodeNetwork()

        zero_bias = np.zeros(1, )
        y = np.array([1 / 2])

        nn.append_layer(nodetemplate.ReLU(1), np.array([[1]]), zero_bias)
        nn.append_layer(nodetemplate.ReLU(1), np.array([[2]]), zero_bias)

        sut.nn = nn

        (actual_weights, actual_biases, actual_activations) = \
            sut.step(inputs.onen, y, costs.dabs_squared)

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_weights, "w")

        expected = [np.array([3]), np.array([6])]
        self.assertEqual(expected, actual_biases, "b")

        expected = [np.array([6]), np.array([6])]
        self.assertEqual(expected, actual_activations, "a")

    def test_backprop_shape_3layer_node(self):
        sut = NodeModel()

        nn = NodeNetwork()

        zero_bias = np.zeros((4,))
        y = np.zeros((4,))
        id_weight =  np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
        ])

        nn.append_layer(nodetemplate.ReLU(4, input_shape=4), id_weight, zero_bias)
        nn.append_layer(nodetemplate.ReLU(4, input_shape=4), id_weight, zero_bias)

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

    def test_backprop_conv_node(self):
        np.random.seed()
        N = 5
        x = np.arange(N**2).reshape((N, N))

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True),
            nodetemplate.Sigmoid(2),
        ])

        sut.nn.weights = [
            np.array(
                [
                [ [[0.16687128]], [ [1.2187011]],  [ [0.51149002]]],
                [ [[0.9165428]], [[ -1.15340098]], [[-0.35499248]]],
                [ [[0.32682511]], [[-0.13196826]], [[-0.80919688]]]
                ]
            ),

            np.array(
                [[-0.96587943, -0.95681104, -0.18343715, -0.9363369,   1.15871117, -0.25671794, -0.3784069,   0.7668324,
                  0.26413521, -1.12501609, -0.12716973,  0.16934818, 0.55445104,
                  0.92922783, -1.02276463, -0.81728719, -0.04095724, -0.93718699, 0.38794503, -0.42924279,  0.20760751,
                  0.2761578, -0.97354242, -0.63260593, 0.6840177],

                 [-0.04584017, -0.23713031,  0.22691451, -0.14610676,  0.4875827, -0.5654631,
                  -0.52868807, -0.7257997, -0.96755309, -1.22175077,  1.17440455,  0.53355628, -1.00281615,  0.55518733,
                  0.30208984, -0.68935598, -0.05261074,  0.15993541, 0.77265757,  1.16255869, -0.56024119, -1.13546629,
                  0.07811296,  0.83475526, -0.58973099]]
            ),
        ]

        sut.nn.biases = [
            np.array([0.11756599]),
            np.array( [0.90301752, 0.4992944 ])
        ]

        numericW, numericB = numeric_grad(sut, x)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared)

        gradW.reverse()

        i = 0
        for numeric, auto in zip(numericW, gradW):
            # print(f"Numeric:")
            # print(numeric[abs(numeric) > 10**-3])
            # print(f"Auto:")
            # print(auto[abs(auto) > 10 ** -3])
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=3, msg=f"Layer {i}")
            i += 1

        gradB.reverse()

        for numeric, auto in zip(numericB, gradB):
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=3)

    def test_backprop_conv_linear(self):
        np.random.seed(15)
        N = 5
        x = np.arange(N**2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True),
            nodetemplate.Linear(10),
            nodetemplate.ReLU(10),
            nodetemplate.Linear(3),
            nodetemplate.ReLU(3),
            nodetemplate.Sigmoid(3),
        ])

        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))

        numericW, numericB = numeric_grad(sut, x)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)

        sut.predict(x, plot_activations=False)

        numericW.reverse()

        layer = len(numericW) - 1
        for numeric, auto in zip(numericW, gradW):
            # print(f"Numeric:")
            # print(numeric[abs(numeric) > 10 ** -3])
            # print(f"Auto:")
            # print(auto[abs(auto) > 10 ** -3])
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer -= 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            # print(auto[abs(auto) > 10 ** -3])
            print(auto)
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            print("")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multifilters(self):
        np.random.seed()
        N = 5
        x = np.arange(N ** 2).reshape((N, N))

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True),
            nodetemplate.ReLU(2),
        ])

        numericW, numericB = numeric_grad(sut, x)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)

        gradW.reverse()

        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10**-3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=0)

        numericB.reverse()

        for numeric, auto in zip(numericB, gradB):
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=0)

    def test_backprop_multiconv_std_params(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(15)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            # libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape, flatten_output=True),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.ReLU(1),
        ])

        sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        sut.nn.weights[2] = np.ones((1, 25))

        sut.nn.biases[0] = np.zeros(1)
        sut.nn.biases[1] = np.zeros(1)
        sut.nn.biases[2] = np.zeros([1])

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)

        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multiconv_rand_wparams(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(23)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.ReLU(1),
        ])

        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[2] = np.ones((1, 25))

        sut.nn.biases[0] = np.ones(1)
        sut.nn.biases[1] = np.ones(1)
        sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape, "\n")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multiconv_rand_bparams(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(23)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.ReLU(1),
        ])

        sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        sut.nn.weights[2] = np.ones((1, 25))

        # sut.nn.biases[0] = np.ones(1)
        # sut.nn.biases[1] = np.ones(1)
        # sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape, "\n")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1


    def test_backprop_multiconv_rand_params(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(22)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.ReLU(10),
        ])

        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[2] = np.ones((1, 25))

        # sut.nn.biases[0] = np.ones(1)
        # sut.nn.biases[1] = np.ones(1)
        # sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape, "\n")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multiconv_rand_params2(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(22)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.Linear(10),
            nodetemplate.ReLU(10),
            nodetemplate.Sigmoid(10),
        ])

        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[2] = np.ones((1, 25))

        # sut.nn.biases[0] = np.ones(1)
        # sut.nn.biases[1] = np.ones(1)
        # sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape, "\n")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multiconv_rand_params3(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(21)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()

        sut.build([
            libs.model.templates.conv2d.Conv2D(F=5, P=2, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1),
            libs.model.templates.conv2d.Conv2D(F=5, P=2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1),
            libs.model.templates.conv2d.Conv2D(F=5, P=2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.Linear(10),
            nodetemplate.ReLU(10),
            nodetemplate.Sigmoid(10),
        ])

        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[2] = np.ones((1, 25))

        # sut.nn.biases[0] = np.ones(1)
        # sut.nn.biases[1] = np.ones(1)
        # sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            print(f"Auto mean")
            print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape, "\n")
            self.assertEqual(numeric.shape, auto.shape)
            self.assertAlmostEqual((numeric - auto).sum(), 0, places=1, msg=f"Layer {layer}")
            layer += 1

    def test_backprop_multifilterconv(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(22)
        N = 5
        x = np.arange(N ** 2).reshape((N, N)) / np.arange(N**2).sum()

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=1, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=4),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=8),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=8, flatten_output=True),
            # libs.model.templates.conv2d.Conv2D(F=3, P=1, K=8, flatten_output=True),
            # libs.model.templates.conv2d.Conv2D(F=3, P=1, K=32, flatten_output=True),
            # libs.model.templates.conv2d.Conv2D(F=3, P=1, flatten_output=True),
            nodetemplate.Linear(5),
            nodetemplate.ReLU(5),
        ])


        # sut.nn.weights[0] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[1] = np.ones((3, 3, 1, 1))
        # sut.nn.weights[2] = np.ones((1, 25))

        # sut.nn.biases[0] = np.ones(1)
        # sut.nn.biases[1] = np.ones(1)
        # sut.nn.biases[2] = np.ones(1)

        numericW, numericB = numeric_grad(sut, x)
        sut.predict(x, plot_activations=False)
        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            error: np.ndarray = abs((numeric - auto) / (numeric + 1/10**8))
            expected_error = abs((numeric - numeric*.05) / (numeric + 1/10**8))
            diff = expected_error - error
            diff[diff >= 0] = True

            # print(expected_error)
            # print(error)
            # print(diff)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertTrue(diff.all())

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            print(f"Numeric:")
            print(numeric[abs(numeric) > 10 ** -3])
            print(f"Auto:")
            print(auto[abs(auto) > 10 ** -3])
            # print(f"Auto mean")
            # print((auto[abs(auto) > 10 ** -3]).mean())
            print(numeric.shape, auto.shape)
            error: np.ndarray = abs((numeric - auto) / (numeric + 1 / 10 ** 8))
            expected_error = abs((numeric - numeric * .05) / (numeric + 1 / 10 ** 8))
            diff = expected_error - error
            diff[diff >= 0] = True

            # print(expected_error)
            # print(error)
            # print(diff)
            self.assertEqual(numeric.shape, auto.shape)
            self.assertTrue(diff.all())

            layer += 1

    def test_backprop_mnist_conv(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(22)

        (x_train, y_train), (_, _) = datasets.load_mnist()
        x: np.ndarray = x_train[0]
        y: np.ndarray = y_train[0]

        xprocessed = x / np.amax(x)

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=32, input_shape=x.shape),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=32),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=4, flatten_output=True),
            nodetemplate.Linear(5, c=0.2),
            nodetemplate.ReLU(5),
            nodetemplate.Linear(10, c=0.05),
            nodetemplate.Sigmoid(10),
        ])

        sut.predict(xprocessed, plot_activations=False)

    def test_backprop_maxpool(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(24)
        N = 6
        shape = (N, N)
        x = np.random.rand(*shape)
        x = (x / x.sum())

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=1, input_shape=x.shape),
            libs.model.templates.maxpool2d.MaxPool2D(3, 3, flatten_output=True),
            nodetemplate.ReLU(5)
        ])

        numericW, numericB = numeric_grad(sut, x)
        prediction = sut.predict(x, plot_activations=False)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            cmp = cmp_arr(numeric, auto)
            self.assertTrue(cmp["cmp_shape"])
            self.assertTrue(cmp["cmp_val"])

            layer += 1

        numericB.reverse()

        layer = 0
        print("Bias")
        for numeric, auto in zip(numericB, gradB):
            cmp = cmp_arr(numeric, auto)
            self.assertTrue(cmp["cmp_shape"])
            self.assertTrue(cmp["cmp_val"])

            layer += 1

    def test_backprop_maxpool2(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(24)
        N = 28
        shape = (N, N)
        x = np.random.rand(*shape)
        x = (x / x.sum())

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, input_shape=x.shape),
            libs.model.templates.maxpool2d.MaxPool2D(2, 2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1),
            libs.model.templates.maxpool2d.MaxPool2D(2, 2, flatten_output=True),
            nodetemplate.Linear(10),
            nodetemplate.ReLU(10),
            nodetemplate.Linear(10),
        ])

        numericW, numericB = numeric_grad(sut, x)
        prediction = sut.predict(x, plot_activations=False)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print("Layer", layer)
            cmp = cmp_arr(numeric, auto)
            self.assertTrue(cmp["cmp_shape"])
            self.assertTrue(cmp["cmp_val"])

            layer += 1

        numericB.reverse()

    def test_backprop_maxpool3(self):
        np.set_printoptions(precision=2, suppress=True)
        np.random.seed(25)
        N = 28
        shape = (N, N)
        x = np.random.rand(*shape)
        x = (x / x.sum())

        sut = NodeModel()
        sut.build([
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=2, input_shape=x.shape),
            libs.model.templates.maxpool2d.MaxPool2D(2, 2),
            libs.model.templates.conv2d.Conv2D(F=3, P=1, K=4),
            libs.model.templates.maxpool2d.MaxPool2D(2, 2, flatten_output=True),
            nodetemplate.Linear(10),
            nodetemplate.ReLU(10),
            nodetemplate.Linear(10),
        ])

        numericW, numericB = numeric_grad(sut, x)
        prediction = sut.predict(x, plot_activations=False)

        gradW, gradB, gradA = sut.step(x, np.array(0), costs.dabs_squared, plot_w_grad=False)


        numericW.reverse()
        layer = 0
        for numeric, auto in zip(numericW, gradW):
            print("Layer", len(numericW) - layer - 1)

            if auto is None:
                print("SKIP")
                layer += 1
                continue

            cmp = cmp_arr(numeric, auto)
            self.assertTrue(cmp["cmp_shape"], f"Mismatching shapes NUMERIC {numeric.shape} AUTO {auto.shape}")
            self.assertTrue(cmp["cmp_val"])

            layer += 1

        numericB.reverse()


if __name__ == '__main__':
    unittest.main()
