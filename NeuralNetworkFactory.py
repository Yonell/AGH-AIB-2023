import NeuralNetwork
import numpy as cp
import numpy as np
import ActivationFunction as af
import random
import pickle


class NeuralNetworkFactory:
    def __init__(self):
        return

    @staticmethod
    def create_random_neural_network(layer_count, layers_units_count, activation_fn=None):
        if layer_count is None:
            raise ValueError("layer_count must be provided")
        if layers_units_count is None:
            raise ValueError("layers_units_count must be provided")
        if len(layers_units_count) != layer_count:
            raise ValueError("layers_units_count must be the same length as layer_count")
        weights = []
        biases = []
        if activation_fn is None:
            activation_fn = af.ActivationFunction(
                af.availible_activation_functions[random.randint(0, len(af.availible_activation_functions) - 1)])
        for i in range(1, layer_count):
            weights.append(np.random.normal(0, 1, (layers_units_count[i - 1], layers_units_count[i])))
            biases.append(np.random.normal(0, 1, layers_units_count[i]))
        return NeuralNetwork.NeuralNetwork(layer_count, layers_units_count, weights, biases, activation_fn)

    @staticmethod
    def create_neural_network_from_parents(parent1, parent2):
        if parent1 is None:
            raise ValueError("Parent1 must be provided")
        if parent2 is None:
            raise ValueError("Parent2 must be provided")
        if parent1.layer_count != parent2.layer_count:
            raise ValueError("Parents must have the same number of layers")
        if parent1.layers_units_count != parent2.layers_units_count:
            raise ValueError("Parents must have the same number of units in each layer")
        nn = NeuralNetwork.NeuralNetwork(init_from_parents=True)
        nn.init_from_parents(parent1, parent2)
        return nn

    @staticmethod
    def save_neural_network_to_file(nn, file_path):
        if nn is None:
            raise ValueError("Neural network must be provided")
        if file_path is None:
            raise ValueError("File path must be provided")
        with open(file_path, 'wb') as f:
            pickle.dump(nn, f)
            f.close()
        return

    @staticmethod
    def load_neural_network_from_file(file_path):
        if file_path is None:
            raise ValueError("File path must be provided")
        with open(file_path, 'rb') as f:
            nn = pickle.load(f)
            f.close()
        return nn
