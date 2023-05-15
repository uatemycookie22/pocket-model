class LayerTemplate:
    # current_n is layer L
    # prev_n is layer L-1
    def __init__(self, current_n, prev_n, activator, dactivator):
        self.n = prev_n
        self.k = current_n
        self.activator = activator
        self.dactivator = dactivator # derivative of activator


