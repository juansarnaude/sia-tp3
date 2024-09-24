import random

import numpy as np

from src.models.Neuron import Neuron


class PerceptronStep:

    def __init__(self, learning_rate, periods, epsilon, dataset):
        self.learning_rate = learning_rate
        self.periods = periods
        self.epsilon = epsilon
        self.dataset = dataset

    def run(self):
        current_period = 1
        current_error = 100
        bias = random.uniform(0,1)

        w_amount = len(self.dataset.iloc[0]) - 1
        print(w_amount)
        neuron = Neuron(w_amount)

        while current_period < self.periods and current_error > self.epsilon:
            expected_values = []
            computed_values = []

            for data in self.dataset.values.tolist():
                expected_value = data.pop()

                neuron_weighted_sum = neuron.get_weighted_sum(data, bias)
                neuron_theta = neuron.theta(neuron_weighted_sum)
                neuron_computed = neuron.compute_activation(neuron_theta)

                new_ws = np.zeros(w_amount)
                for index, w in enumerate(neuron.get_w()):
                    new_ws[index] = w + self.delta_w(neuron_computed, expected_value, data[index])

                neuron.modify_w(new_ws)

                bias += self.delta_b(neuron_computed, expected_value)

                expected_values.append(expected_value)
                computed_values.append(neuron_computed)

            print("Current Period: " + str(current_period))
            print(expected_values)
            print(computed_values)

            if self.error(np.array(computed_values), np.array(expected_values)) <= self.epsilon:
                print("Last Period WON")
                return

            current_period += 1


    def delta_w(self, neuron_computed, expected_value, data):
        return self.learning_rate * (expected_value - neuron_computed) * data

    def delta_b(self, neuron_computed, expected_value):
        return self.learning_rate * (expected_value - neuron_computed)

    def error(self, computed, expected):
        error_acum = 0
        for index in range(len(computed)):
            error_acum += abs(expected[index] - computed[index])
        return error_acum
