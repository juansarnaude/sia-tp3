from abc import ABC
import random
import numpy as np

from src.models.Neuron import Neuron
from src.plot.hiperplane import graph_hiperplane

class Perceptron(ABC):

    def __init__(self, w_amount, learning_rate):
        self.bias = random.uniform(-0.1, 0.1)
        self.w_amount = w_amount
        self.neuron = Neuron(self.w_amount)
        self.learning_rate = learning_rate
        self.min_value_in_dataset = None
        self.max_value_in_dataset = None

    def run(self, periods, epsilon, dataset):
        periods = periods
        epsilon = epsilon
        dataset = dataset

        current_period = 1
        current_error = 100

        #Initialize the normalization
        for data in dataset.values.tolist():
            value = data.pop()
            if not self.min_value_in_dataset:
                self.min_value_in_dataset = value
                self.max_value_in_dataset = value
            self.min_value_in_dataset = min(self.min_value_in_dataset, value)
            self.max_value_in_dataset = max(self.max_value_in_dataset, value)


        while current_period < periods and current_error > epsilon:
            expected_values = []
            computed_values = []
            data_list = dataset.values.tolist()

            for data in data_list: # TODO optimze this so it doesnt have to load data this way always
                expected_value = data.pop()

                neuron_weighted_sum = self.neuron.get_weighted_sum(data, self.bias)
                theta = self.theta(neuron_weighted_sum)
                computed_value = self.compute_activation(theta)

                self.neuron.update_w( self.delta_w(computed_value, expected_value, np.array(data), neuron_weighted_sum))

                self.bias += self.delta_b(computed_value, expected_value, neuron_weighted_sum)

                expected_values.append(expected_value)
                computed_values.append(computed_value)

            print("Current Period: " + str(current_period))
            # (neuron.get_w())
            print(expected_values)
            print(computed_values)

            error = self.error(np.array(computed_values), np.array(expected_values))
            print("Error: " + str(error))

            if error <= epsilon:
                print("Last Period WON")
                return
            
            weights = self.neuron.weights

            #graph_hiperplane(weights[0], weights[1], self.bias)

            current_period += 1


    def value_to_feed_forward(self, neuron_input):
        neuron_weighted_sum = self.neuron.get_weighted_sum(neuron_input, self.bias)
        theta = self.theta(neuron_weighted_sum)
        return self.compute_activation(theta)

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

    def theta_diff(self, h):
        pass

    def normalize(self, value):
        if not -1 <= value <= 1:
            raise ValueError(f"El valor {value} no es entre 0 y 1.")
        #return value
        return np.interp(value, [-1, 1], [self.min_value_in_dataset, self.max_value_in_dataset])