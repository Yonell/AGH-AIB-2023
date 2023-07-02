import copy

import ModelDoodles
import ModelDoodlesFiles
import random

from NeuralNetwork import NeuralNetwork
import multiprocessing as mp


class GeneticAlgorithmTrainer:
    def __init__(self, population_size=120, layers_count=5, layers_units_count=None, side=32, activation_fn=None):
        if layers_units_count is None:
            layers_units_count = [32 * 32, 1000, 500, 400, 16]
        self.files = ModelDoodlesFiles.ModelDoodlesFiles(image_size=side)
        self.population_size = population_size
        self.layers_count = layers_count
        self.layers_units_count = layers_units_count
        self.population = []
        self.best_ever_nn = None
        self. best_ever_fitness = float("-inf")
        for i in range(0, population_size):
            self.population.append(ModelDoodles.ModelDoodles(self.files, layers_count, layers_units_count, False,
                                                             side=side, activation_fn=activation_fn))
        return

    def calculate_fitness_of_one(self, i):
        self.population[i].get_fitness()

        return

    def calculate_fitness_async(self):
        pool = mp.Pool(mp.cpu_count())
        pool.map(self.calculate_fitness_of_one, range(0, len(self.population)))
        pool.close()
        return

    def iterate(self, mutation_rate=0.0001):
        # print("Sorting...")
        for i in range(0, len(self.population)):
            # print("Calculating fitness of doodle " + str(i) + " of " + str(len(self.population)))
            self.population[i].get_fitness()
        self.population.sort(key=lambda x: x.get_fitness())
        # print("Cutting population...")
        self.population = self.population[int(self.population_size / 3):]
        # print("New population size: " + str(len(self.population)))
        # print("Creating new population...")
        random.shuffle(self.population)
        for i in range(0, int(self.population_size / 3)):
            self.population.append(
                ModelDoodles.ModelDoodles(self.files, self.layers_count, self.layers_units_count, True))
            self.population[len(self.population) - 1].init_from_parents(self.population[2 * i],
                                                                        self.population[2 * i + 1])
        # print("New population size: " + str(len(self.population)))
        # print("Mutating...")
        for i in range(0, len(self.population)):
            # print("Mutating doodle " + str(i) + " of " + str(len(self.population)))
            self.population[i].mutate(mutation_rate)
        # print("Sorting...")
        for i in range(0, len(self.population)):
            # print("Calculating fitness of doodle " + str(i) + " of " + str(len(self.population)))
            self.population[i].get_fitness()
        self.population.sort(key=lambda x: x.get_fitness())
        if self.best_ever_fitness <= self.population[len(self.population) - 1].get_fitness():
            self.best_ever_fitness = self.population[len(self.population) - 1].get_fitness()
            self.best_ever_nn = NeuralNetwork(self.layers_count, self.layers_units_count,
                                              self.population[len(self.population) - 1].get_weights(),
                                              self.population[len(self.population) - 1].get_biases(),
                                              self.population[len(self.population) - 1].nn.activation_fn)
        return

    def get_best(self):
        for i in range(0, len(self.population)):
            self.population[i].get_fitness()
        self.population.sort(key=lambda x: x.get_fitness())
        return self.population[len(self.population) - 1]

    def get_best_ever(self):
        return self.best_ever_nn

    def get_best_ever_fitness(self):
        return self.best_ever_fitness
