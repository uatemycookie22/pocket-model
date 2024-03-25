from libs.model.model import Model, read
import numpy as np
from libs.model.layertemplate import ReLU, Sigmoid, Linear
from libs.utils import datasets, dataset_processors as dp

READ_MODEL = True

if __name__ == '__main__':
    print("Seeding...")
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data loading
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    # Preprocessing
    x_flat_n = x_test.shape[1] * x_test.shape[2]
    x_test = dp.norm(dp.grayscale(dp.flatten_2d(x_test)))
    y_test = dp.one_hot_encode(y_test, 10)

    sut: Model | None = None

    if READ_MODEL:
        # Model construction
        postfix = '05-19|19:17'
        sut = read(f'models/model.json')
    else:
        sut = Model()
        sut.build([
            Linear(24, x_flat_n, c=0.2),
            ReLU(24),
            Linear(10, c=0.05),
            Sigmoid(10),
        ])

    pred_len = len(x_test)
    sut.eval(x_test[:pred_len], y_test[:pred_len], plot=True)

