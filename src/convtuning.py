import datetime

from libs.model.node_model import read
from libs.model.node_model import NodeModel
from libs.model.nodetemplate import ReLU, Sigmoid, Linear
from libs.model.templates.conv2d import Conv2D
from libs.model.templates.maxpool2d import MaxPool2D
import numpy as np
from libs.utils import datasets, dataset_processors as dp
import convtunetrain

from libs.utils import datasets

import multiprocessing

if __name__ == "__main__":
    # Define the number of processes (cores) to use
    num_processes = multiprocessing.cpu_count()

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Define the arguments for your function
    arguments = range(6)


    # Define your function to be executed
    # Map the function to the arguments using the pool of processes
    results = pool.map(convtunetrain.tune_conv_lr, arguments)

    # Close the pool to free resources
    pool.close()
    pool.join()
