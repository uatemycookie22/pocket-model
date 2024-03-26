import time

import numpy as np
import skorch.scoring
import torch
import torch.nn as nn
from skorch import NeuralNet
from skorch.callbacks import LRScheduler, PrintLog, EpochTimer, ProgressBar, TensorBoard
from torch import optim
from torch.nn import Conv2d, MaxPool2d, ReLU
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from libs.utils.datasets import load_mnist_fashion

import libs.utils.dataset_processors as dp

# from skorch.callbacks import Freezer
# freezer = Freezer(lambda x: not x.startswith('model.fc'))
# from skorch.callbacks import Checkpoint
# checkpoint = Checkpoint(
#     f_params='best_model.pt', monitor='valid_acc_best')


# Data selection
(x_train, y_train), (x_test, y_test) = load_mnist_fashion()

train_size = 12 * 1000 * 5
x_train = x_train[:train_size].reshape(train_size, 1, 28, 28).astype(np.float32)
y_train = y_train[:train_size]

x_test = x_test.reshape(len(x_test), 1, 28, 28).astype(np.float32)
# Preprocessing
x_flat_n = x_train.shape[1] * x_train.shape[2]
x_shape = x_train[0].shape

x_train = dp.zero_center(dp.grayscale(x_train))
y_train = dp.one_hot_encode(y_train, 10).astype(np.float32)

x_test = dp.zero_center(dp.grayscale(x_test))
y_test = dp.one_hot_encode(y_test, 10).astype(np.float32)



model = torch.nn.Sequential(
    Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1,),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    nn.BatchNorm2d(num_features=32),
    Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    ReLU(),
    Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.BatchNorm1d(64 * 3 * 3),
    nn.Linear(64 * 3 * 3, 128),  # out_channels * W/4 * W/4
    nn.ReLU(),
    nn.Linear(128, 10), # Assuming output size for MNIST classification
    nn.Softmax(dim=1),
)

net = NeuralNet(
    model,
    criterion=nn.MSELoss(reduction='sum'),
    max_epochs=5*10, # 5 epochs per 2 minutes
    lr=10**-3,
    batch_size=16,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    train_split=None,
    optimizer=optim.SGD,
    optimizer__momentum=0.9,
    callbacks=[
        LRScheduler(policy='StepLR', step_size=5, gamma=0.5),
        PrintLog(),
        ProgressBar(),
        EpochTimer(),
        TensorBoard(writer=SummaryWriter())
    ],

)

net.fit(x_train, y_train)

correct = 0
total_cost = 0
total_rt = 0
results = []

preds = net.predict_proba(x_test)
pred_len = len(preds)

total_cost = skorch.scoring.loss_scoring(net, x_test, y_test)
average_cost = total_cost / len(x_test)

correct = (preds.argmax(axis=1) == y_test.argmax(axis=1)).sum()


stats = {
    "accuracy": correct / pred_len,
    "average_cost": total_cost / pred_len,
}


print(f"Accuracy: {'%0.2f' % (correct/pred_len)} ({correct}/{pred_len})")
print(f"Average cost: {'%0.2f' % average_cost}")


