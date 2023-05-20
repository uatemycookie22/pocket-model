Toy deep learning project to play with perceptrons. Can only optimize using SGD.

Example code:
```python
from libs.model.model import Model
import numpy as np
from libs.model.layertemplate import ReLU, Sigmoid, Linear
from libs.utils import datasets, dataset_processors as dp

if __name__ == '__main__':
    np.random.seed()

    # Data selection
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()

    train_size = 12*1000*5
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    # Preprocessing
    x_flat_n = x_train.shape[1] * x_train.shape[2]
    x_train = dp.norm(dp.grayscale(dp.flatten_2d(x_train)))
    y_train = dp.one_hot_encode(y_train, 10)

    # Model construction
    sut = Model()
    sut.build([
        Linear(30, x_flat_n, c=0.2),
        ReLU(30),
        Linear(10, c=0.05),
        Sigmoid(10),
    ])

    # Accuracy before
    x_test = dp.norm(dp.grayscale(dp.flatten_2d(x_test)))
    y_test = dp.one_hot_encode(y_test, 10)
    pred_len = 1000
    sut.eval(x_test[:pred_len], y_test[:pred_len])

    # Train
    print("Start training...")
    batch_size = 24
    learning_rate = 3
    sut.train(
        x_train,
        y_train,
        m=batch_size,
        l=learning_rate,
        plot_cost=True,
        epochs=1,  # 1 epoch gets to around 80% accuracy
    )

    # Accuracy after
    sut.eval(x_test[:pred_len], y_test[:pred_len])

```

After running the example code, the loss curve and the accuracy curve will be plotted.

<img width="441" alt="image" src="https://github.com/uatemycookie22/pocket-model/assets/67762761/03235f21-4882-45a0-bdf1-fbff48fea50d">
