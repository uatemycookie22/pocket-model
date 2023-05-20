from libs.model_helpers import activators


class LayerTemplate:
    # current_n is layer L
    # prev_n is layer L-1
    def __init__(self, layer_name: str, n: int, prev_n: int, activator, dactivator):
        self.layer_name = layer_name
        self.n = n
        self.prev_n = prev_n
        self.activator = activator
        self.dactivator = dactivator # derivative of activator


class ReLU(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        self.layer_name = 'relu'

        super().__init__(self.layer_name, current_n, prev_n, activators.relu, activators.drelu)


class LeakyReLu(LayerTemplate):
    def __init__(self, current_n, prev_n=0, alpha=0.2):
        self.layer_name = 'leaky_relu'
        self.alpha = alpha

        super().__init__(
            self.layer_name,
            current_n,
            prev_n,
            lambda x: activators.leaky_relu(x, alpha),
            lambda x: activators.dleaky_relu(x, alpha)
        )


class Sigmoid(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        self.layer_name = 'sigmoid'

        super().__init__(self.layer_name, current_n, prev_n, activators.sigmoid, activators.dsigmoid)


class Tanh(LayerTemplate):
    def __init__(self, current_n, prev_n=0):
        self.layer_name = 'tanh'

        super().__init__(self.layer_name, current_n, prev_n, activators.tanh, activators.dtanh)


class Linear(LayerTemplate):
    def __init__(self, current_n, prev_n=0, c=1):
        self.layer_name = 'linear'
        self.c = c

        super().__init__(
            self.layer_name,
            current_n,
            prev_n,
            lambda x: activators.linear(x, c),
            lambda x: activators.dlinear(x, c)
        )


class Normal(LayerTemplate):
    def __init__(self, current_n, prev_n=0, c=1):
        super().__init__(
            self.layer_name,
            current_n,
            prev_n,
            lambda x: activators.linear(x, c),
            lambda x: activators.dlinear(x, c)
        )


