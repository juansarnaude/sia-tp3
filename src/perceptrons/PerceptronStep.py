from src.perceptrons.Perceptron import Perceptron


class PerceptronStep(Perceptron):

    def __init__(self, learning_rate, periods, epsilon, dataset):
        super().__init__(learning_rate, periods, epsilon, dataset)


    def delta_w(self, neuron_computed, expected_value, data):
        return self.learning_rate * (expected_value - neuron_computed) * data

    def delta_b(self, neuron_computed, expected_value):
        return self.learning_rate * (expected_value - neuron_computed)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)):
            error_acum += abs(expected[index] - computed[index])
        return error_acum

    def theta(self,weighted_sum):
        if weighted_sum >= 0:
            return 1
        return -1

    def compute_activation(self, theta_value):
        if theta_value > 0:
            return 1
        return -1

