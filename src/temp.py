# import multiprocessing
#
# import numpy as np
#
#
# def compute(x, y):
#     return (x ** 2) - y
#
# # Define a function that will be executed in parallel
# def worker_function(x, y, result_list):
#     result_list.put(compute(x, y))
#
# if __name__ == "__main__":
#     # Number of CPU cores
#     num_processes = multiprocessing.cpu_count()
#
#     # Shared memory list to store results
#     x = multiprocessing.Value(10)
#     y = multiprocessing.Value(1)
#     result = multiprocessing.Queue()
#
#     # Create processes
#     processes = []
#     for i in range(num_processes):
#         process = multiprocessing.Process(target=worker_function, args=(x, y, result))
#         processes.append(process)
#         process.start()
#
#     # Wait for all processes to finish
#     for process in processes:
#         process.join()
#
#     # Print results
#     while not result.empty():
#         print(result.get())
import multiprocessing
import time
import random

from multiprocessing import Process, Queue, current_process, freeze_support

from libs.utils.process_queue import ProcessQueue


#
# Function run by worker processes
#

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)


#
# Functions referenced by tasks
#

def mul(a, b):
    return a * b


def plus(a, b):
    return a + b


if __name__ == '__main__':
    freeze_support()
    pq = ProcessQueue()
    for i in range(10):
        pq.submit(mul, 2, 2)

    pq.start()

    time.sleep(1)
    result = pq.get()

    print(result)

    pq.stop()
