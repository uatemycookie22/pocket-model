## Dependencies
- Numpy
- Tensorflow (purely for the keras datasets)
- Matplotlib

## Description
Toy deep learning project to play with perceptrons. Can only optimize using SGD + Momentum.

Example code:
```python
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Softmax
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import dataset_processors as dp
from libs.utils import datasets

if __name__ == '__main__':
    np.random.seed()
    np.set_printoptions(precision=2, suppress=True)

    # Data selection
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist_fashion()

    x_shape = x_train[0].shape

    train_size = 12 * 1000 * 5
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_train = dp.zero_center(dp.grayscale(x_train))
    y_train = dp.one_hot_encode(y_train, 10)

    x_test = dp.zero_center(dp.grayscale(x_test))
    y_test = dp.one_hot_encode(y_test, 10)

    # Model construction
    sut = NodeModel()
    sut.build([
        Conv2D(
            F=5,                 # Filter size; height and width (usually F=3 or F=5)
            P=2,                 # Padding (usually 1 for F=3 and 2 for F=5)  
            K=4,                 # Number of filters (usually powers of 2, e.g 4, 8, 16, wouldn't recommend over 16)  
            input_shape=x_shape  # Shape of the input (only necessary for the first layer)              
        ),
        MaxPool2D(F=2,           # Maxpool usually F=2 and S=2
                  S=2),          # S=Stride
        Conv2D(F=3, P=1, K=8),
        MaxPool2D(F=2, S=2, flatten_output=True),
        ReLU(64),
        Softmax(10),             # Softmax good for classifying multiple classes
    ])

    # Accuracy before
    pred_len = 100
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    print("Start")
    sut.train(
        x_train,
        y_train,
        m=12,               # Batch size (keep below 12 if multiprocessing)
        l=0.1,              # Learning rate
        epochs=1,
        p_progress=0.1,     # How often to update training accuracy (0 to 1)
        rho=0.9,            # Momentum decay
        plot_cost=True,     # Plot the loss & accuracy curve vs time
        momentum=True,      # SGD + Momentum
        multiprocess=True,  # Use multiple CPU cores for one batch
    )

    # Accuracy after
    pred_len = 1000
    sut.eval(x_test[:pred_len], y_test[:pred_len], plot=True)
```

After running the example code, the loss curve and the accuracy curve will be plotted.

<img width="441" alt="image" src="https://github.com/uatemycookie22/pocket-model/assets/67762761/03235f21-4882-45a0-bdf1-fbff48fea50d">

# Multiprocessing
To enable multiprocessing, set `NodeModel().train(multiprocessing=True)`. There is currently an issue with inaccurate 
gradients occurring with batch sizes greater than 12, so for now set `m=12` or lower in `NodeModel().train(m=12)`.

# Layers

Currently, these are the defined layers that are available out of the box:
- libs.model.templates.conv2d.Conv2D
- libs.model.templates.maxpool2d.MaxPool2D
- libs.model.nodetemplate.ReLu
- libs.model.nodetemplate.Softmax
- libs.model.nodetemplate.Sigmoid
- libs.model.nodetemplate.TanH

# Planned additions
Layers:
- Batch Normalization
- Dropout
- Regularization

Optimizers:
- Adam

Plotting:
- Confusion matrix