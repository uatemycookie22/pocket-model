import datetime

from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp

from libs.utils import datasets

READ_MODEL = True

if __name__ == '__main__':
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    # Model construction
    postfix = '03_23_1548'

    sut: NodeModel | None = read(f'../models/model_{postfix}.json')

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 10000
    sut.eval(x_test[:pred_len], y_test[:pred_len], print_preds=True, plot=True)
