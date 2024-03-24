import datetime

from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp

from libs.utils import datasets
import multiprocessing

SAVE_MODEL = False
READ_MODEL = False

if __name__ == '__main__':
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    train_size = 12 * 1000 * 1
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_shape = x_train[0].shape
    x_train = dp.zero_center(dp.grayscale(x_train))
    y_train = dp.one_hot_encode(y_train, 10)

    sut: NodeModel | None = None

    # Model construction
    postfix = '03_24_1029'

    if READ_MODEL:
        sut = read(f'models/model_{postfix}.json')
    else:
        sut = NodeModel()
        sut.build([
            # Conv2D(F=3, P=1, K=1, input_shape=x_shape),
            # MaxPool2D(F=2, S=2),
            # Conv2D(F=3, P=1, K=1),
            # MaxPool2D(F=2, S=2, flatten_output=True),
            ReLU(128),
            Sigmoid(10)
        ])

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    result = sut.train(x_train, y_train, m=16, l=0.1, momentum=False, plot_cost=True, epochs=30, p_progress=0.1)

    # Accuracy after
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])


    if SAVE_MODEL:
        sut.save('./models/model', postfix_timestamp=True)
