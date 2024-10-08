import numpy as np

from src.perceptrons.Perceptron import Perceptron

class PerceptronLinear(Perceptron):
    def __init__(self, w_amount, learning_rate,epsilon,out_file):
        super().__init__(w_amount, learning_rate,epsilon,out_file)


    def delta_w(self, neuron_computed, expected_value, data, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum) * data # TODO : TOTO (by Africa) IS REALLY 1 ??

    def delta_b(self, neuron_computed, expected_value, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)):
            error_acum += (expected[index] - computed[index])**2
        return error_acum / len(computed)

    def theta(self,weighted_sum):
        return weighted_sum

    def compute_activation(self, theta_value):
        return theta_value

    def theta_diff(self, h):
        return 1