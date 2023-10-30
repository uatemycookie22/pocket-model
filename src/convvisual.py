from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp

from libs.utils import datasets

SAVE_MODEL = False
READ_MODEL = False

if __name__ == '__main__':
    print("Seeding...")
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
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
    postfix = '10_29_0250'

    if READ_MODEL:
        sut = read(f'models/model_{postfix}.json')
    else:
        sut = NodeModel()
        sut.build([
            # Conv2D(F=5, P=2, K=8, input_shape=x_shape),
            # Conv2D(F=5, P=2, K=16, flatten_output=True),
            Conv2D(F=5, P=2, K=16, input_shape=x_shape, flatten_output=True),
            Linear(24, c=0.5),
            ReLU(24),
            Linear(10, c=0.5),
            Sigmoid(10),
        ])

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    sut.predict(x_train[0], plot_activations=True)


    # sut.train(x_train, y_train, m=12 * 2, l=10, plot_cost=True, epochs=1)
