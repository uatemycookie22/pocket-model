from libs.utils import datasets
import matplotlib.pyplot as plt
import numpy as np


def show_set(num_samples, x, y):
    fig, axes = plt.subplots(num_samples // 3 + 1, 3, figsize=(12, 3))

    for i in range(num_samples):
        image = x[i].reshape((28, 28))
        axes[i // 3, i % 3].imshow(image, cmap='gray')  # Use proper indexing for subplots
        axes[i // 3, i % 3].set_title(f"Label: {y[i]}")
        axes[i // 3, i % 3].axis('off')

    plt.tight_layout()
    plt.show()


def show_mnist(num_samples=9):
    (x_train, y_train), (_, _) = datasets.load_mnist()

    show_set(num_samples, x_train, y_train)


def show_mnist_rand(num_samples=9):
    np.random.seed()

    (x_train, y_train), (_, _) = datasets.load_mnist()

    rand_indices = np.random.choice(len(x_train), size=num_samples, replace=False)

    rand_x = x_train[rand_indices]
    rand_y = y_train[rand_indices]

    show_set(num_samples, rand_x, rand_y)

