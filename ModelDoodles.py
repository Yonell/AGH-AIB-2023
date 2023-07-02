import NeuralNetwork
import numpy as np
import NeuralNetworkFactory as nnf
import pandas as pd
import random
import multiprocessing as mp


class ModelDoodles:
    def __init__(self, files, layers_count=None, layers_units_count=None, special_init=False, activation_fn=None,
                 side=128):
        if files is None:
            raise ValueError("files must be provided")
        self.files = files
        if special_init:
            return
        if layers_count is None:
            raise ValueError("layers_count must be provided")
        if layers_units_count is None:
            raise ValueError("layers_units_count must be provided")
        if len(layers_units_count) != layers_count:
            raise ValueError("layers_units_count must be the same length as layers_count")
        if layers_count < 2:
            raise ValueError("layers_count must be at least 2")
        if layers_units_count[0] != side * side:
            raise ValueError("layers_units_count[0] must be " + str(side * side))
        if layers_units_count[layers_count - 1] != len(self.files.model_doodles_categories) + 1:
            raise ValueError(
                "layers_units_count[layers_count-1] must be " + str(len(self.files.model_doodles_categories) + 1))
        self.side = side
        self.nn = nnf.NeuralNetworkFactory().create_random_neural_network(layers_count, layers_units_count,
                                                                          activation_fn=activation_fn)
        self.fitness = self.calculate_fitness()
        return

    def init_from_parents(self, parent1, parent2):
        self.nn = nnf.NeuralNetworkFactory().create_neural_network_from_parents(parent1.nn, parent2.nn)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self, samples_count=64):
        self.fitness = 0
        for i in range(0, samples_count):
            sample_set = random.randint(0, len(self.files.model_doodles_samples) - 1)
            sample = self.files.model_doodles_samples[sample_set][
                random.randint(0, self.files.model_doodles_samples[sample_set].shape[0] - 1)]
            result = self.nn.calculate_value(sample)
            result -= np.max(result)  # softmax
            result = np.exp(result)
            result = result / np.sum(result)
            if sample_set < len(self.files.model_doodles_categories):
                r_2 = result.copy()
                r_2[sample_set] = 1 - r_2[sample_set]
                np.square(r_2, r_2)
                self.fitness -= np.sum(r_2)
            else:
                r_2 = result.copy()
                r_2[len(self.files.model_doodles_categories)] = 1 - r_2[len(self.files.model_doodles_categories)]
                np.square(r_2, r_2)
                self.fitness -= np.sum(r_2)
            pass
        self.fitness /= samples_count
        return self.fitness

    def mutate(self, mutation_rate=0.001):
        self.fitness = None
        self.nn.mutate(mutation_rate)

    def get_fitness(self):
        if self.fitness == float("-inf"):
            self.calculate_fitness()
        if self.fitness is None:
            self.calculate_fitness()
        return self.fitness

    def get_weights(self):
        return self.nn.get_weights()

    def get_biases(self):
        return self.nn.get_biases()
