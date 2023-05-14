from libs.model_helpers import activators
import matplotlib.pyplot as plt
import numpy as np


def show_relu(x=np.linspace(-10, 10, 100)):
    z = activators.relu(x)
    plt.plot(x, z)
    plt.xlabel('x')
    plt.ylabel('ReLu(x)')
    plt.show()


def show_tanh(x=np.linspace(-10, 10, 100)):
    z = activators.tanh(x)
    plt.plot(x, z)
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.show()


def show_sigmoid(x=np.linspace(-10, 10, 100)):
    z = activators.sigmoid(x)
    plt.plot(x, z)
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.show()

