import json
import numpy as np
from libs.model import layertemplate


def ndarray_dump(weight_list: list[np.ndarray], output_file):
    # Convert numpy arrays to lists for JSON serialization
    weight_list = [weight.tolist() for weight in weight_list]

    with open(output_file, 'w') as f:
        json.dump(weight_list, f)


def json_read_ndarray(input_file):
    with open(input_file, 'r') as f:
        weight_list = json.load(f)

    return [np.array(weights) for weights in weight_list]


def model_dump(output: dict, output_file: str):

    with open(output_file, 'w') as f:
        json.dump(output, f)


def model_read(input_file: str) -> dict:
    model_input: dict | None = None
    with open(input_file, 'r') as f:
        model_input = json.load(f)

    if not model_input:
        raise IOError(f"Could not read {input_file}")

    return model_input


def layer_factory(layer_name: str, **kwargs) -> layertemplate.LayerTemplate:
    layer_nodes = kwargs.get('n')
    prev_nodes = kwargs.get('prevN') or 0
    linear_constant = kwargs.get('c') or 1
    alpha = kwargs.get('alpha') or 0.2

    match layer_name:
        case 'relu':
            return layertemplate.ReLU(layer_nodes, prev_nodes)
        case 'linear':
            return layertemplate.Linear(layer_nodes, prev_nodes, linear_constant)
        case 'sigmoid':
            return layertemplate.Sigmoid(layer_nodes, prev_nodes)
        case 'leaky_relu':
            return layertemplate.LeakyReLu(layer_nodes, prev_nodes, alpha)
        case 'tanh':
            return layertemplate.Tanh(layer_nodes, prev_nodes)
        case _:
            raise ValueError(f'{layer_name} not a valid layer identifier')





