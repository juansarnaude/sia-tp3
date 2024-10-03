import numpy as np

from src.perceptrons.Perceptron import Perceptron

class PerceptronNonLinear(Perceptron):
    def __init__(self, w_amount, learning_rate, beta=8):
        super().__init__(w_amount, learning_rate)
        self.beta = beta

    def delta_w(self, neuron_computed, expected_value, data, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum) * data # TODO : TOTO (by Africa) IS REALLY 1 ??

    def delta_b(self, neuron_computed, expected_value, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * self.theta_diff(neuron_weighted_sum)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)-1):
            error_acum += (expected[index] - computed[index])**2
        print(error_acum)
        return 1/2 * error_acum                                                     # TODO check if its divided by 2 or by the len(computed[0])

    def theta(self,weighted_sum):
        return weighted_sum

    def compute_activation(self, h):
        # return np.tanh(self.beta*h)
        return 1/(1+np.exp(-2*self.beta*h))
    
    def theta_diff(self, h):
        # return self.beta*(1-(self.compute_activation(h)**2))
        return 2*self.beta*self.compute_activation(h)*(1-self.compute_activation(h))