import numpy as np

from src.perceptrons.Perceptron import Perceptron

class PerceptronLinear(Perceptron):
    def __init__(self, learning_rate, periods, epsilon, dataset):
        super().__init__(learning_rate, periods, epsilon, dataset)


    def delta_w(self, neuron_computed, expected_value, data):
        return self.learning_rate * (expected_value - neuron_computed) * 1 * data # TODO : TOTO (by Africa) IS REALLY 1 ??

    def delta_b(self, neuron_computed, expected_value):
        return self.learning_rate * (expected_value - neuron_computed)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)-1):
            error_acum += (expected[index] - computed[index])**2
        print(error_acum)
        return 1/2 * error_acum                                                     # TODO check if its divided by 2 or by the len(computed[0])

    def theta(self,weighted_sum):
        return weighted_sum

    def compute_activation(self, theta_value):
        return theta_value