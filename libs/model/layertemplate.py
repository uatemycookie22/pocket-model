from libs.model_helpers import activators


class LayerTemplate:
    # current_n is layer L
    # prev_n is layer L-1
    def __init__(self, n: int, prev_n: int, activator, dactivator):
        self.n = n
        self.prev_n = prev_n
        self.activator = activator
        self.dactivator = dactivator # derivative of activator


class ReLU(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        super().__init__(current_n, prev_n, activators.relu, activators.drelu)


class Sigmoid(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        super().__init__(current_n, prev_n, activators.sigmoid, activators.dsigmoid)


class Tanh(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        super().__init__(current_n, prev_n, activators.tanh, activators.dtanh)

