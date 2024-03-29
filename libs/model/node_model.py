import math
import multiprocessing
import time
from typing import Any

import numpy as np
from libs.model import nodetemplate
from libs.model import layertemplate
from libs.model.network import Network
from libs.model.node_network import NodeNetwork
from libs.model_helpers import linalg
from libs.model_helpers import costs
from libs.plotters.model_plots import CostRT, Eval, ActivationLog, WeightGradLog
from libs.utils import io
from datetime import datetime
import datetime as dt
import concurrent.futures
import itertools
import multiprocessing as mp

from libs.utils.process_queue import ProcessQueue

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)


def step_worker(a, label: np.ndarray, dcost_func, weights, biases, layer_templates):
    activations = [a]
    for b, w, activator in zip(biases, weights, layer_templates):
        a = activator.f(a, w, b)
        activations.append(a)

    dc_dw = []
    dc_db = []
    dc_da = []

    dc_daL = None
    for L in range(len(weights) - 1, -1, -1):
        layer: Any = layer_templates[L]
        w = weights[L]
        b = biases[L]

        # Find derivative of cost with respect to its weights
        if dc_daL is None:
            dc_daL = dcost_func(activations[L + 1], label)

        dc_dwL, dc_dbL, dc_daL = layer.from_upstream(dc_daL, activations[L], w, b)

        dc_dw.append(dc_dwL)
        dc_da.append(dc_daL)
        dc_db.append(dc_dbL)

    return dc_dw, dc_db, dc_da


class NodeModel:
    def __init__(self, costf=costs.abs_squared, dcostf=costs.dabs_squared):
        self.pq = None
        self.nn = NodeNetwork()
        self.costf = costf
        self.dcostf = dcostf
        self._built = False
        self.activationLog = ActivationLog()
        self.w_gradLog = WeightGradLog()
        self.multiprocess = False

        # Momentum
        self.vx = []

    # Given a list of layers (n nodes for layer L and activator),
    # build randomized neural network
    def build(self, layers: list[nodetemplate.NodeTemplate]):
        assert self._built is not True, "_built must be False"
        for layer in layers:
            self.nn.from_layer(layer)

        self._built = True

    def predict(self, x, plot_activations=False):
        activations = self.nn.feed_forwards(x)

        if plot_activations:
            self.activationLog.add(activations)
            self.activationLog.plot()

        return activations[-1]

    def sample(self, input: np.ndarray, label: np.ndarray, cost_func):
        a = self.nn.feed_forward(input)
        return np.sum(cost_func(a, label))

    def step(self, input: np.ndarray, label: np.ndarray, dcost_func, plot_w_grad=False):
        activations = self.nn.feed_forwards(input)

        dc_dw = []
        dc_db = []
        dc_da = []

        dc_daL = None
        for L in range(len(self.nn.weights) - 1, -1, -1):
            layer: Any = self.nn.layer_templates[L]
            w = self.nn.weights[L]
            b = self.nn.biases[L]

            # Find derivative of cost with respect to its weights
            if dc_daL is None:
                dc_daL = dcost_func(activations[L + 1], label)

            dc_dwL, dc_dbL, dc_daL = layer.from_upstream(dc_daL, activations[L], w, b)

            dc_dw.append(dc_dwL)
            dc_da.append(dc_daL)
            dc_db.append(dc_dbL)

        if plot_w_grad:
            self.w_gradLog.add(dc_dw)
            self.w_gradLog.plot()

        return dc_dw, dc_db, dc_da

    def batch(self, x: list[np.ndarray], y: list[np.ndarray]):
        weight_gradients = []
        bias_gradients = []


        if self.multiprocess is True:
            for x_sample, y_sample in zip(x, y):
                # total_cost += sut.sample(x, y, costs.abs_squared)
                weights_clone = self.nn.weights[:]
                biases_clone = self.nn.biases[:]
                templates_clone = self.nn.layer_templates[:]

                self.pq.submit(step_worker, x_sample, y_sample, self.dcostf, weights_clone, biases_clone,
                               templates_clone)

            batch_grads = self.pq.get()

            for grads in batch_grads:
                dc_dw, dc_db, _ = grads
                weight_gradients.append(dc_dw)
                bias_gradients.append(dc_db)
        else:
            for x_sample, y_sample in zip(x, y):
                # total_cost += sut.sample(x, y, costs.abs_squared)
                dc_dw, dc_db, _ = self.step(x_sample, y_sample, self.dcostf)

                weight_gradients.append(dc_dw)
                bias_gradients.append(dc_db)

        avg_weight_gradients = linalg.avg_gradient(weight_gradients)
        avg_bias_gradients = linalg.avg_gradient(bias_gradients)

        avg_weight_gradients.reverse()
        avg_bias_gradients.reverse()

        return avg_weight_gradients, avg_bias_gradients

    def train(self, train_x: np.ndarray, train_y: np.ndarray,
              epochs=1, m=12, l=0.003, plot_cost=False, plot_accuracy=False, quit_threshold=0.01,
              plot_w_grad=False, p_progress=0.1, rho=0.9, momentum=False, multiprocess=False):
        plotter = CostRT()

        plotter.plot()
        train_len = len(train_y)

        self.vx = []  # Initialize momentum

        if multiprocess:
            # Initialize multiprocessing
            print("Starting processes...")
            multiprocessing.freeze_support()
            manager = multiprocessing.Manager()
            task_queue, done_queue = manager.Queue(), manager.Queue()

            self.multiprocess = True
            self.pq = ProcessQueue(min(m, 12), task_queue, done_queue)
            self.pq.start()

        train_start = time.time()
        for epoch in range(epochs):
            train_x, train_y = linalg.shuffle(train_x, train_y)

            if plot_cost:
                print(f"\nEpoch {epoch}")
                print("Duration\tProgress\tBatch rt\tTrain accuracy\tTraining cost\n"
                      "_______________________________________________________________________")
            stats = None
            init_stats = None
            for i in range(0, train_len, m):
                batch_start = time.time()
                batch_sample_x = train_x[i:i + m]
                batch_sample_y = train_y[i:i + m]

                w_gradients, b_gradient = self.batch(batch_sample_x, batch_sample_y)
                batch_rt = time.time() - batch_start

                # Update momentum
                if len(self.vx) == 0:
                    self.vx = w_gradients
                else:
                    for L in range(len(self.vx)):
                        self.vx[L] = rho * self.vx[L] + w_gradients[L]

                # Update weights
                if momentum:
                    for L in range(len(w_gradients)):
                        self.nn.weights[L] -= self.vx[L] * (l / m)
                        self.nn.biases[L] -= b_gradient[L] * (l / m)
                else:
                    for L in range(len(w_gradients)):
                        self.nn.weights[L] -= w_gradients[L] * (l / m)
                        self.nn.biases[L] -= b_gradient[L] * (l / m)

                # Evaluation
                if plot_cost:
                    batch = i // m
                    modulo = max(1, (math.floor(train_len * p_progress) // m))
                    progress = "%0.2f" % (100 * i / train_len)

                    if batch % modulo == 0:
                        stats = self.eval(train_x[:19], train_y[:19], summary=False)
                        init_stats = stats if init_stats is None else init_stats
                        plotter.add(stats['average_cost'], stats['accuracy'])

                    if batch % 10 == 0:
                        if stats is None:
                            print(f"\r{'%0.2f' % (time.time() - train_start)}\t\t"
                                  f"{progress}%\t\t"
                                  f"{'%0.4f' % batch_rt}\t"
                                  , end='', flush=True)
                        else:
                            print(f"\r{'%0.2f' % (time.time() - train_start)}\t\t"
                                  f"{progress}%\t\t"
                                  f"{'%0.2f' % (1000 * batch_rt)}\t\t"
                                  f"{'%0.2f' % (init_stats['accuracy'])} -> {'%0.2f' % (stats['accuracy'])}\t"
                                  f"{'%0.2f' % (init_stats['average_cost'])} -> {'%0.2f' % (stats['average_cost'])}\t"
                                  , end='', flush=True)

                if plot_w_grad:
                    self.w_gradLog.add(w_gradients)

            print("")
        if plot_cost:
            plotter.show()

        train_time = time.time() - train_start
        duration = str(dt.timedelta(seconds=int(train_time)))
        print(f"\n\nEvaluation\n---------------\nTotal train time: {duration} hh:mm:ss")

        if self.multiprocess:
            self.pq.flush()
            self.pq.stop()

        return {
            "train_time": train_time
        }

    def eval(self, x, y, summary=True, print_preds=False, plot=False):
        assert len(x) == len(y), "Length of training data must be same as length of labels"
        pred_len = len(x)
        correct = 0
        total_cost = 0
        total_rt = 0
        results = []
        for sample_x, sample_y in zip(x, y):
            start = time.time()
            preds = self.predict(sample_x)

            cost = self.sample(sample_x, sample_y, cost_func=self.costf)

            total_cost += cost
            total_rt += time.time() - start

            sample_results = {
                "cost": cost,
                "x": sample_x,
                "y": preds,
                "prediction": preds.argmax(),
                "label": sample_y.argmax(),
            }

            results.append(sample_results)

            if print_preds:
                print(f"Prediction: {preds} {preds.argmax()} Label: {sample_y.argmax()}")

            if preds.argmax() == sample_y.argmax():
                correct += 1

        stats = {
            "accuracy": correct / pred_len,
            "average_cost": total_cost / pred_len,
            "average_ff_rt": total_rt / pred_len,
        }

        if summary:
            print(f"Accuracy: {'%0.2f' % stats['accuracy']} ({correct}/{pred_len})")
            print(f"Average cost: {'%0.2f' % stats['average_cost']}")
            # print(f"Average ff time: {'%0.2f' % stats['average_ff_rt']}")

        if plot:
            eval_plotter = Eval()
            eval_plotter.show_top_cost(results)
            eval_plotter.show_low_incorrect(results)
            eval_plotter.show_low_correct(results)

        return stats

    def save(self, file_name="params", postfix_timestamp=False):
        if postfix_timestamp:
            current_datetime = datetime.now()
            postfix = current_datetime.strftime("%m_%d_%H%M")
            file_name = f"{file_name}_{postfix}"

        weight_list = [np.round(weight, decimals=3).tolist() for weight in self.nn.weights]
        bias_list = [np.round(bias, decimals=3).tolist() for bias in self.nn.biases]
        layers_list = []

        for layer in self.nn.layer_templates:
            clean_dict = {}
            for key, val in layer.__dict__.items():
                if not io.is_jsonable(val):
                    continue

                clean_dict[key] = val

            layers_list.append(clean_dict)

        output = {
            "weights": weight_list,
            "biases": bias_list,
            "layers": layers_list,
        }

        io.model_dump(output, f"{file_name}.json")


def read(file_name: str) -> NodeModel:
    model_input = io.model_read(file_name)
    weights = [np.array(weights) for weights in model_input['weights']]
    biases = [np.array(biases) for biases in model_input['biases']]
    layers = [io.nodelayer_factory(**layer) for layer in model_input['layers']]

    model = NodeModel()
    model.build(layers)
    model.nn.weights = weights
    model.nn.biases = biases

    return model
