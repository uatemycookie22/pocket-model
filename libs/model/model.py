import time
import numpy as np
from libs.model.layertemplate import LayerTemplate
from libs.model import layertemplate
from libs.model.network import Network
from libs.model_helpers import linalg
from libs.model_helpers import costs
from libs.plotters.model_plots import CostRT, Eval
from libs.utils import io
from datetime import datetime


class Model:
    def __init__(self, costf = costs.abs_squared, dcostf = costs.dabs_squared):
        self.nn = Network()
        self.costf = costf
        self.dcostf = dcostf
        self._built = False
        self._tcosts: list[int] = []

    # Given a list of layers (n nodes for layer L and activator),
    # build randomized neural network
    def build(self, layers: list[LayerTemplate]):
        assert self._built is not True, "_built must be False"
        for layer in layers:
            self.nn.appendr_layer(layer)
        self._built = True

    def sample(self, input: np.ndarray, label: np.ndarray, cost_func):
        a = self.nn.feed_forward(input)
        return np.sum(cost_func(a, label))

    def step(self, input: np.ndarray, label: np.ndarray, dcost_func):
        activations = self.nn.feed_forwards(input)

        dc_dw = []
        dc_db = []
        dc_da = []

        dc_daL = None
        for L in range(len(self.nn.weights) - 1, -1, -1):
            w = self.nn.weights[L]  # w(L): j X k
            b = self.nn.biases[L]  # b(L): j X 1
            z = np.dot(w, activations[L]) + b  # z(L): j X 1
            a = activations[L+1]  # a(L): j X 1

            # Find derivative of cost with respect to its weights
            if dc_daL is None:
                dc_daL = dcost_func(a, label) # j X 1

            dc_da_dz = linalg.dcost_db(dc_daL, self.nn.dactivators[L](z))
            dc_db.append(dc_da_dz)  # j X 1
            dc_dw.append(linalg.dcost_dw(dc_da_dz, activations[L]))  # j X k
            dc_daL = linalg.dcost_dpreva(dc_da_dz, w)  # j X 1
            dc_da.append(dc_daL)  # j X 1

        return dc_dw, dc_db, dc_da

    def batch(self, x: list[np.ndarray], y: list[np.ndarray]):
        weight_gradients = []
        bias_gradients = []

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
              epochs=1, m=12, l=0.003, plot_cost=False, plot_accuracy=False, quit_threshold=0.01):
        plotter = CostRT()
        if plot_cost:
            plotter.plot()
        train_len = len(train_y)

        for epoch in range(epochs):
            # Shuffle
            data_train = list(zip(train_x, train_y))
            np.random.shuffle(data_train)
            train_x, train_y = zip(*data_train)
            train_x, train_y = np.array(train_x), np.array(train_y)

            for i in range(0, train_len, m):
                batch_start = time.time()
                batch_sample_x = train_x[i:i + m]
                batch_sample_y = train_y[i:i + m]

                # Calculate gradient
                w_gradients, b_gradient = self.batch(batch_sample_x, batch_sample_y)
                batch_rt = time.time() - batch_start

                # Update weights
                for L in range(len(w_gradients)):
                    self.nn.weights[L] -= w_gradients[L] * (l / m)
                    self.nn.biases[L] -= b_gradient[L] * (l / m)

                # Evaluation
                stats = self.eval(train_x[:100], train_y[:100], summary=False)

                if (i // m) % max(1, ((train_len*10) // (100 * m))) == 0 and i > 0:
                    progress = "%0.2f" % (100 * i / train_len)
                    print(f"Epoch: {epoch}\n"
                          f"Progress: {progress}%\n"
                          f"Train cost: {'%0.2f' % (stats['average_cost'])}\n"
                          f"Train accuracy: {'%0.2f' % stats['accuracy']}\n"
                          f"Batch rt: {'%0.2f' % batch_rt}")
                if plot_cost:
                    plotter.add(stats['average_cost'], stats['accuracy'])

        plotter.show()

    def predict(self, x):
        return self.nn.feed_forward(x)

    def save(self, file_name="params", postfix_timestamp=False):
        if postfix_timestamp:
            current_datetime = datetime.now()
            postfix = current_datetime.strftime("%m-%d|%H:%M")
            file_name = f"{file_name}_{postfix}"

        weight_list = [weight.tolist() for weight in self.nn.weights]
        bias_list = [bias.tolist() for bias in self.nn.biases]
        layers_list = [
            {
                "layerName": layer.layer_name,
                "n": layer.n,
                "prevN": layer.prev_n,
                "c": layer.c if isinstance(layer, layertemplate.Linear) else None,
                "alpha": layer.alpha if isinstance(layer, layertemplate.LeakyReLu) else None,
            }
            for layer in self.nn.layer_templates
        ]

        output = {
            "weights": weight_list,
            "biases": bias_list,
            "layers": layers_list,
        }

        io.model_dump(output, f"{file_name}.json")

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
            print(f"Accuracy: { '%0.2f' % stats['accuracy']} ({correct}/{pred_len})")
            print(f"Average cost: {'%0.2f' % stats['average_cost']}")
            # print(f"Average ff time: {'%0.2f' % stats['average_ff_rt']}")

        if plot:
            eval_plotter = Eval()
            eval_plotter.show_top_cost(results)
            eval_plotter.show_low_incorrect(results)
            eval_plotter.show_low_correct(results)

        return stats


def read(file_name: str) -> Model:
    model_input = io.model_read(file_name)
    weights = [np.array(weights) for weights in model_input['weights']]
    biases = [np.array(biases) for biases in model_input['biases']]
    layers = [io.layer_factory(layer['layerName'], **layer) for layer in model_input['layers']]

    model = Model()
    model.build(layers)
    model.nn.weights = weights
    model.nn.biases = biases

    return model
