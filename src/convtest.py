from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
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

    train_size = 12 * 100 * 1
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_shape = x_train[0].shape
    x_train = dp.norm(x_train)
    y_train = dp.one_hot_encode(y_train, 10)

    sut: NodeModel | None = None

    # Model construction
    postfix = '10_30_0056'

    if READ_MODEL:
        sut = read(f'models/model_{postfix}.json')
    else:
        sut = NodeModel()
        sut.build([
            Conv2D(F=3, P=1, K=4, input_shape=x_shape),
            MaxPool2D(F=2, S=2),
            Conv2D(F=3, P=1, K=8),
            MaxPool2D(F=2, S=2, flatten_output=True),
            # Conv2D(F=5, P=2, K=16, input_shape=x_shape, flatten_output=True),
            Linear(128),
            ReLU(128),
            Linear(10),
            Sigmoid(10)
        ])

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    print(f"Mean: {x_train.mean()} Stdev: {x_train.std()}")
    print("Start")
    sut.train(x_train, y_train, m=16, l=1, plot_cost=True, epochs=1, p_progress=0.25)

    # Accuracy after
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])


    if SAVE_MODEL:
        sut.save('./models/model', postfix_timestamp=True)
