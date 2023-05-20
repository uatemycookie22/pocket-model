from libs.model.model import Model, read
import numpy as np
from libs.model.layertemplate import ReLU, Sigmoid, Linear
from libs.utils import datasets, dataset_processors as dp

SAVE_MODEL = True
READ_MODEL = True

if __name__ == '__main__':
    print("Seeding...")
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    train_size = 12*1000*5
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_train = dp.norm(dp.grayscale(dp.flatten_2d(x_train)))
    y_train = dp.one_hot_encode(y_train, 10)

    sut: Model | None = None

    # Model construction
    postfix = '05-19|19:17'

    if READ_MODEL:
        sut = read(f'models/model_{postfix}.json')
    else:
        sut = Model()
        sut.build([
            Linear(24, x_flat_n, c=0.2),
            ReLU(24),
            Linear(10, c=0.05),
            Sigmoid(10),
        ])

    x_test = dp.norm(dp.grayscale(dp.flatten_2d(x_test)))
    y_test = dp.one_hot_encode(y_test, 10)

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    print("Start")
    sut.train(x_train, y_train, m=12*2, l=3, plot_cost=True, epochs=1)

    # Accuracy after
    pred_len = 1000
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    if SAVE_MODEL:
        sut.save('models/model', postfix_timestamp=True)
