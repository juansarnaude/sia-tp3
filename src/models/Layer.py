
import numpy as np

from src.perceptrons.PerceptronLinear import PerceptronLinear
from src.perceptrons.PerceptronNonLinear import PerceptronNonLinear
from src.perceptrons.PerceptronStep import PerceptronStep


class Layer:
    def __init__(self, perceptron_amount, layer_input_size, learning_rate):
        self.perceptron_list = []
        self.last_computed_output = []   # Saves the value returned on last computation
        self.last_theta_diff = []

        for i in range(perceptron_amount):
            # We initialize all the unused params in 0.
            self.perceptron_list.append(PerceptronLinear(layer_input_size, learning_rate))

    def feed_forward(self, layer_input):
        layer_output = []

        for perceptron in self.perceptron_list:
            layer_output.append(perceptron.value_to_feed_forward(layer_input))

        self.last_computed_output = layer_output
        return layer_output

    def get_theta_diff(self):
        theta_diff_list = []
        for perceptron in self.perceptron_list:
            theta_diff_list.append(perceptron.theta_diff(perceptron.neuron.last_weighted_sum))
        self.last_theta_diff = theta_diff_list
        return theta_diff_list

    def get_w_list(self):
        w_list = []
        for perceptron in self.perceptron_list:
            w_list.append(perceptron.neuron.get_w())
        return np.array(w_list)

    def update_w(self, delta_w_layer):
        index = len(self.perceptron_list)-1 # Capaz hay que iterar al revez
        for delta_perceptron in delta_w_layer:
            self.perceptron_list[index].neuron.update_w(delta_perceptron)
            print(delta_perceptron)
            index-=1
        print("------------------------------")