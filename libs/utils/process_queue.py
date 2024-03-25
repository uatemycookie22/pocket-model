import multiprocessing
from multiprocessing import Process


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)

        output.put(result)


class ProcessQueue:
    def __init__(self, n, task_queue, done_queue):
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.done_queue = self.manager.Queue()
        self.n = n
        self.jobs = []

    def submit(self, func, *args):
        self.task_queue.put((func, args))

    def start(self):
        for i in range(self.n):
            process = Process(target=worker, args=(self.task_queue, self.done_queue))
            process.start()
            self.jobs.append(process)

    def get(self):
        # Get and print results
        results = []
        for i in range(self.n):
            results.append(self.done_queue.get())

        return results

    def flush(self):
        for i in range(self.n):
            self.task_queue.put('STOP')

    def stop(self):

        for job in self.jobs:
            job.join()
