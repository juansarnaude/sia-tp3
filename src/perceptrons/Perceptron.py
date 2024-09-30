from abc import ABC
import random
import numpy as np

from src.models.Neuron import Neuron

class Perceptron(ABC):

    def __init__(self, learning_rate, periods, epsilon, dataset):
        self.learning_rate = learning_rate
        self.periods = periods
        self.epsilon = epsilon
        self.dataset = dataset

    def run(self):
        current_period = 1
        current_error = 100
        bias = random.uniform(-0.1, 0.1)
        #bias = 1
        w_amount = len(self.dataset.iloc[0]) - 1
        neuron = Neuron(w_amount)

        while current_period < self.periods and current_error > self.epsilon:
            expected_values = []
            computed_values = []
            data_list = self.dataset.values.tolist()

            for data in data_list: # TODO optimze this so it doesnt have to load data this way always
                expected_value = data.pop()

                neuron_weighted_sum = neuron.get_weighted_sum(data, bias)
                theta = self.theta(neuron_weighted_sum)
                computed_value = self.compute_activation(theta)

                # print(neuron_weighted_sum)
                # print(theta)
                # print(expected_value)
                neuron.update_w( self.delta_w(computed_value, expected_value, np.array(data), neuron_weighted_sum))
                # print(neuron.get_w())
                # print(bias)

                bias += self.delta_b(computed_value, expected_value, neuron_weighted_sum)

                expected_values.append(expected_value)
                computed_values.append(computed_value)

            print("Current Period: " + str(current_period))
            # (neuron.get_w())
            print(expected_values)
            print(computed_values)

            if self.error(np.array(computed_values), np.array(expected_values)) <= self.epsilon:
                print("Last Period WON")
                return

            current_period += 1


    # TODO maybe some of this methods is equal in all perceptrons, need to double check
    def delta_w(self, neuron_computed, expected_value, data, neuron_weighted_sum):
        pass

    def delta_b(self, neuron_computed, expected_value, neuron_weighted_sum):
        pass

    def error(self, computed, expected):
        pass

    def theta(self, weighted_sum):
        pass

    def compute_activation(self, theta_value):
        pass