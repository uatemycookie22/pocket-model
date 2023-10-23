from libs.model.model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Conv2D, Sigmoid, Linear
import numpy as np
from libs.utils import datasets, dataset_processors as dp

from libs.utils import datasets

SAVE_MODEL = True
READ_MODEL = False

if __name__ == '__main__':
    print("Seeding...")
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    train_size = 12 * 10 * 1
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_shape = x_train[0].shape
    x_train = dp.zero_center(dp.grayscale(x_train))
    y_train = dp.one_hot_encode(y_train, 10)

    sut: NodeModel | None = None

    # Model construction
    postfix = '10-22|20:28'

    if READ_MODEL:
        pass
        # sut = read(f'models/model.json')
    else:
        sut = NodeModel()
        sut.build([
            Conv2D(F=3, P=1, flatten_output=True, input_shape=x_shape),
            Linear(24, c=0.2),
            ReLU(24),
            Linear(10, c=0.05),
            Sigmoid(10),
        ])

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    print("Start")
    sut.train(x_train, y_train, m=12 * 2, l=20, plot_cost=True, epochs=1)

    # Accuracy after
    pred_len = 1000
    sut.eval(x_test[:pred_len], y_test[:pred_len])


    if SAVE_MODEL:
        sut.save('./models/model', postfix_timestamp=True)
