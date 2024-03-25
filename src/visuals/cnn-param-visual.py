import datetime

from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp
from libs.plotters.model_plots import FilterColor

from libs.utils import datasets

READ_MODEL = True

if __name__ == '__main__':
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    # Model construction
    postfix = '03_23_1426'

    sut: NodeModel | None = read(f'../models/model_{postfix}.json')

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len], print_preds=True, plot=True)

    # Weight plotting
    weight_plotter = FilterColor()
    x_ones = []
    y_ones = []
    for i in range(100):
        x = x_test[i]
        y = y_test[i]

        if y.argmax() == 1:
            x_ones.append(x)
            y_ones.append(y)

    for i in range(10):
        test_sample = x_ones[i]
        activations = sut.nn.feed_forwards(test_sample)[1:]

        for weights, layer, activation in zip(sut.nn.weights, sut.nn.layer_templates, activations):
            if layer.layer_name == "conv2d":
                # unrolled_filters = [weights[:, :, :, i] for i in range(weights.shape[3])]
                weight_plotter.plot_ndarrays([activation[:, :, i] for i in range(activation.shape[2])])

    # weight_plotter.plot_ndarrays([sut.])

    # sut.build([
    #     Conv2D(F=3, P=1, K=4, input_shape=x_shape),
    #     MaxPool2D(F=2, S=2),
    #     Conv2D(F=3, P=1, K=8),
    #     MaxPool2D(F=2, S=2, flatten_output=True),
    #     ReLU(256),
    #     Sigmoid(10)
    # ])