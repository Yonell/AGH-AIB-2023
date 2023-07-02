import numpy as np


class Layer(object):
    activation_fn = None
    activation_fn_vectorized = None
    weights = None
    bias = None

    def __init__(self, num_units=None, weights=None, bias=None, activation_fn=lambda x: x, prev_layer=None,
                 init_from_parents=False):
        if init_from_parents:
            self.values = None
            self.prev_layer = prev_layer
            self.weights = None
            self.bias = None
            return
        if activation_fn is None and prev_layer is not None:
            raise ValueError("Activation function must be provided")
        if weights is None and prev_layer is not None:
            raise ValueError("Weights must be provided")
        if bias is None:
            bias = np.zeros(num_units)
        if prev_layer is not None and len(bias) != num_units:
            raise ValueError("Bias must be the same length as num_units")
        if prev_layer is not None and len(weights[0]) != num_units:
            raise ValueError("Weights and bias must be the same length")
        self.values = None
        self.prev_layer = prev_layer
        self.num_units = num_units
        if prev_layer is None:
            return
        self.weights = np.array(weights).copy()
        self.bias = np.array(bias).copy()
        self.activation_fn = activation_fn  # przerobić na np.vectorize
        self.activation_fn_vectorized = np.vectorize(activation_fn)
        return

    def init_from_parents(self, parent1, parent2):
        self.weights = (parent1.weights + parent2.weights) / 2
        self.bias = (parent1.bias + parent2.bias) / 2
        self.activation_fn = parent1.activation_fn
        self.activation_fn_vectorized = parent1.activation_fn_vectorized
        self.num_units = parent1.num_units
        return

    def mutate(self, mutation_rate=0.1):
        if self.weights is None:
            raise ValueError("Weights has not been initialized")
        if self.bias is None:
            raise ValueError("Bias has not been initialized")
        values_to_add = np.random.normal(0, mutation_rate, self.weights.shape)
        self.weights += values_to_add
        values_to_add = np.random.normal(0, mutation_rate, self.bias.shape)
        self.bias += values_to_add
        return

    def update_value(self, x=None):
        if x is None:
            if self.prev_layer is None:
                raise ValueError("No input provided")
            x = self.prev_layer.values
        elif len(x) != self.num_units:
            raise ValueError("Input size does not match layer size")
        self.values = self.activation_fn_vectorized((x @ self.weights) + self.bias)
        return

    def set_value(self, x):
        self.values = x
        return

    def set_previous_layer(self, prev_layer):
        self.prev_layer = prev_layer
        return

    def __str__(self):
        return f"Layer:\n {self.num_units} units,\n activation function: {self.activation_fn},\n weights: {self.weights},\n bias: {self.bias}\n"
