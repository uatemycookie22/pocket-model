from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp
from libs.utils import datasets

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

x_test = dp.zero_center(dp.grayscale(x_test))
y_test = dp.one_hot_encode(y_test, 10)

# Define your function to be executed
def tune_conv_lr(arg):
    lr = 0.76
    batch = np.random.randint(1, 32)*2

    sut: NodeModel = NodeModel()
    sut.build([
        Conv2D(F=5, P=2, K=4, input_shape=x_shape),
        MaxPool2D(F=2, S=2),
        Conv2D(F=3, P=1, K=8),
        MaxPool2D(F=2, S=2, flatten_output=True),
        ReLU(256),
        Sigmoid(10)
    ])

    print(f"Learning rate: {'%0.4f' % lr} Batch size: {batch}")

    result = sut.train(x_train, y_train, m=16, l=lr, plot_cost=False, epochs=1, p_progress=0.1)

    # Accuracy after
    pred_len = 100
    stats = sut.eval(x_test[:pred_len], y_test[:pred_len])

    print(f"Learning rate: {'%0.4f' % lr} Batch size: {batch} Accuracy {'%0.2f' % stats['accuracy']}")
