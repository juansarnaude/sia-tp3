import numpy as np

from src.perceptrons.Perceptron import Perceptron
from src.utils import functions


class PerceptronNonLinear(Perceptron):
    def __init__(self, w_amount, learning_rate, beta=8, activation="sigmoid"):
        if activation == "sigmoid":
            self.activation = functions.sigmoid
        elif activation == "tanh":
            self.activation = functions.tanh
        else:
            raise ValueError(f"Not a valid activation function: {activation}.")

        super().__init__(w_amount, learning_rate)
        self.beta = beta

    def delta_w(self, neuron_computed, expected_value, data, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum) * data

    def delta_b(self, neuron_computed, expected_value, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)-1):
            error_acum += (expected[index] - computed[index])**2
        return 1/2 * error_acum

    def theta(self,weighted_sum):
        return weighted_sum

    def compute_activation(self, h):
        # return np.tanh(self.beta*h
        print(self.activation(h))
        return self.normalize(self.activation(h))
    
    def theta_diff(self, h):
        # return self.beta*(1-(self.compute_activation(h)**2))
        return self.activation(h,derivative=True)

    def normalize(self, value):
        if self.activation == "sigmoid":
            return np.interp(value, [0, 1], [self.min_value_in_dataset, self.max_value_in_dataset])
        else:
            return np.interp(value, [-1, 1], [self.min_value_in_dataset, self.max_value_in_dataset])