from src.perceptrons.Perceptron import Perceptron

class PerceptronStep(Perceptron):

    def __init__(self, w_amount, learning_rate):
        super().__init__(w_amount, learning_rate)


    def delta_w(self, neuron_computed, expected_value, data, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed) * data

    def delta_b(self, neuron_computed, expected_value, neuron_weighted_sum):
        return self.learning_rate * (expected_value - neuron_computed)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)-1):
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

    def theta_diff(self, h):
        # This is wrong
        print("this needs to be fixed. Theta diff in perceptronstep.py")
        return 1

