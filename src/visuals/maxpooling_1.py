from libs.model.model import Model, read
import numpy as np
from libs.model.layertemplate import ReLU, Sigmoid, Linear
from libs.model.templates.maxpool2d import MaxPool2D
from libs.utils import datasets, dataset_processors as dp
from libs.model_helpers.linalg import maxpool, unpooling
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Seeding...")
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    train_size = 12*10*1
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_train2 = dp.zero_center(dp.grayscale(x_train.copy()))
    y_train2 = dp.one_hot_encode(y_train, 10)

    # Maxpool
    maxpoolLayer = MaxPool2D(2, 2)


    x_train3 = x_train.copy()
    z3 = maxpoolLayer.f(x_train3[0])
    z4 = maxpoolLayer.from_upstream(np.ones(z3.shape), x_train3[0])[2]

    z, pos = maxpool(x_train3[0], 2, 2)
    z2 = unpooling(z, pos, (28, 28), 2)

    fig, axs = plt.subplots(1, 3, figsize=(12, 12))

    axs[0].imshow(x_train[0], cmap='gray')
    axs[1].imshow(z, cmap='gray')
    axs[2].imshow(z3, cmap='gray')

    plt.tight_layout()
    plt.show()

    x_test = dp.norm(dp.grayscale(dp.flatten_2d(x_test)))
    y_test = dp.one_hot_encode(y_test, 10)

