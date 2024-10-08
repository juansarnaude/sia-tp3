from abc import ABC
import random
import numpy as np

from src.models.Neuron import Neuron
from src.plot.hiperplane import graph_hiperplane

class Perceptron(ABC):

    def __init__(self, w_amount, learning_rate,epsilon,out_file):
        self.bias = random.uniform(-0.1, 0.1)
        self.w_amount = w_amount
        self.neuron = Neuron(self.w_amount)
        self.learning_rate = learning_rate
        self.min_value_in_dataset = None
        self.max_value_in_dataset = None
        self.epsilon = epsilon
        self.out_file = out_file


    def predict(self,data):
        neuron_weighted_sum = self.neuron.get_weighted_sum(data, self.bias)
        theta = self.theta(neuron_weighted_sum)
        return self.compute_activation(theta)


    def train_and_test(self, training_set,training_expected_values,testing_set,testing_expected_values, periods,current_k):
        periods = periods
        epsilon = self.epsilon
        training_set = training_set

        current_period = 1
        current_error = 100

        with open(f"{self.out_file.replace('.csv', '')}{current_k}.csv", 'a') as f:
            f.write(f"period,training_error,testing_error\n")

        # Initialize the normalization
        for value in training_expected_values:
            if not self.min_value_in_dataset:
                self.min_value_in_dataset = value
                self.max_value_in_dataset = value
            self.min_value_in_dataset = min(self.min_value_in_dataset, value)
            self.max_value_in_dataset = max(self.max_value_in_dataset, value)

        while current_period < periods and current_error > epsilon:
            expected_values = []
            computed_values = []
            training_expected_values = training_expected_values

            for i, value in enumerate(training_set):  # TODO optimze this so it doesnt have to load data this way always
                expected_value = training_expected_values[i]
                neuron_weighted_sum = self.neuron.get_weighted_sum(value, self.bias)

                computed_value = self.predict(value)

                self.neuron.update_w(self.delta_w(computed_value, expected_value, np.array(value), neuron_weighted_sum))

                self.bias += self.delta_b(computed_value, expected_value, neuron_weighted_sum)

                expected_values.append(expected_value)
                computed_values.append(computed_value)

            error = self.error(computed_values, expected_values)

            testing_computed_values = []
            for i,testing_value in enumerate(testing_set):
                testing_computed_values.append(self.predict(testing_value))

            testing_error= self.error(testing_computed_values, testing_expected_values)

            with open(f"{self.out_file.replace('.csv', '')}{current_k}.csv", 'a') as f:
                f.write(f"{current_period},{error[0]},{testing_error[0]}\n")

            if error <= epsilon:
                print("Last Period WON")
                print(expected_values)
                print(computed_values)
                return

            weights = self.neuron.weights

            # graph_hiperplane(weights[0], weights[1], self.bias)

            current_period += 1

    def train(self, training_set,training_expected_values, periods, epsilon):
        periods = periods
        epsilon = self.epsilon
        training_set = training_set

        current_period = 1
        current_error = 100

        with open(self.out_file, 'w') as f:
            f.write(f"period,error\n")

        #Initialize the normalization
        for value in training_expected_values:
            if not self.min_value_in_dataset:
                self.min_value_in_dataset = value
                self.max_value_in_dataset = value
            self.min_value_in_dataset = min(self.min_value_in_dataset, value)
            self.max_value_in_dataset = max(self.max_value_in_dataset, value)


        while current_period < periods and current_error > epsilon:
            expected_values = []
            computed_values = []
            training_expected_values = training_expected_values

            for i,value in enumerate(training_set): # TODO optimze this so it doesnt have to load data this way always
                expected_value=training_expected_values[i]
                neuron_weighted_sum = self.neuron.get_weighted_sum(value, self.bias)

                computed_value=self.predict(value)

                self.neuron.update_w( self.delta_w(computed_value, expected_value, np.array(value), neuron_weighted_sum))

                self.bias += self.delta_b(computed_value, expected_value, neuron_weighted_sum)

                expected_values.append(expected_value)
                computed_values.append(computed_value)


            error = self.error(computed_values, expected_values)
            weights = self.neuron.weights

            with open(self.out_file, 'a') as f:
                f.write(f"{current_period},{error[0]}\n")

            if error <= epsilon:
                #graph_hiperplane(weights[0], weights[1], self.bias)
                print("Last Period WON")
                print(expected_values)
                print(computed_values)
                return

            current_period += 1

        #graph_hiperplane(weights[0], weights[1], self.bias)


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
        pass
