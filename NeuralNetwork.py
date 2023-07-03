import numpy as np
import Layer


class NeuralNetwork(object):
    def __init__(self, layer_count=None, layers_units_count=None, weights=None, biases=None, activation_fn=lambda x: x,
                 init_from_parents=False):
        if init_from_parents:
            self.layers = None
            self.layer_count = None
            self.layers_units_count = None
            self.activation_fn = None
            return
        if layer_count is None:
            raise ValueError("layer_count must be provided")
        if layers_units_count is None:
            raise ValueError("layers_units_count must be provided")
        if weights is None:
            raise ValueError("weights must be provided")
        if biases is None:
            raise ValueError("biases must be provided")
        if len(weights) != layer_count-1:
            raise ValueError("weights must be the same length as layer_count")
        if len(biases) != layer_count-1:
            raise ValueError("biases must be the same length as layer_count")
        if len(layers_units_count) != layer_count:
            raise ValueError("layers_units_count must be the same length as layer_count")
        self.layers = [Layer.Layer(layers_units_count[0], None, None, activation_fn, None, False)]
        for i in range(1, layer_count):
            self.layers.append(
                Layer.Layer(layers_units_count[i], weights[i-1], biases[i-1], activation_fn, self.layers[i - 1]))
        self.layer_count = layer_count
        self.layers_units_count = layers_units_count
        self.activation_fn = activation_fn
        return

    def init_from_parents(self, parent1, parent2):
        self.layer_count = parent1.layer_count
        self.layers_units_count = parent1.layers_units_count
        self.activation_fn = [parent1.activation_fn, parent2.activation_fn][np.random.randint(0, 2)]
        self.layers = [Layer.Layer(self.layers_units_count[0], None, None, self.activation_fn, None)]
        for i in range(1, self.layer_count):
            self.layers.append(Layer.Layer(init_from_parents=True))
        for i in range(1, self.layer_count):
            self.layers[i].init_from_parents(parent1.layers[i], parent2.layers[i])
        for i in range(1, self.layer_count):
            self.layers[i].set_previous_layer(self.layers[i - 1])
        return

    def mutate(self, mutation_rate=0.01):
        for i in range(1, self.layer_count):
            self.layers[i].mutate(mutation_rate)
        return

    def calculate_value(self, x=None):
        if x is None:
            raise ValueError("No input provided")
        self.layers[0].set_value(x)
        for i in range(1, self.layer_count):
            self.layers[i].update_value()
        return self.layers[self.layer_count - 1].values

    def calculate_value_with_softmax(self, x=None):
        if x is None:
            raise ValueError("No input provided")
        self.layers[0].set_value(x)
        for i in range(1, self.layer_count):
            self.layers[i].update_value()
        result = self.layers[self.layer_count - 1].values
        result -= np.max(result)
        result = np.exp(result)
        result /= np.sum(result)
        return result

    def __call__(self, x):
        return self.calculate_value(x)

    def __str__(self):
        return ''.join([str(layer) + '\n' for layer in self.layers])

    def get_weights(self):
        return [layer.weights for layer in self.layers[1:]]

    def get_biases(self):
        return [layer.bias for layer in self.layers[1:]]