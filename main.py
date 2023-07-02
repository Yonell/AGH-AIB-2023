import time

import numpy
import numpy as np

import GeneticAlgorithmTrainer as ga
from NeuralNetworkFactory import NeuralNetworkFactory
import ActivationFunction
import NeuralNetwork

if __name__ == '__main__':
    print("Initializing...")
    population = ga.GeneticAlgorithmTrainer(600, 4, [32*32, 50, 20,  16], side=32,
                                            activation_fn=ActivationFunction.ActivationFunction.tanh)
    for i in range(0, 10000000):
        print("Iteration " + str(i) + " starting...")
        start_time = time.time()
        population.iterate(0.00000000001)
        print("Iteration " + str(i) + " fitness: " + str(population.get_best().get_fitness()) + " time: " +
              str(time.time() - start_time))
        print("Average fitness: " + str(np.average([x.get_fitness() for x in population.population])))
        print("Best ever fitness: " + str(population.get_best_ever_fitness()))
        NeuralNetworkFactory().save_neural_network_to_file(population.get_best_ever(), "./best_neural_network.pickle")
    print(population.population[0].get_fitness())
    population.population[0].fitness = None
    exit(0)

