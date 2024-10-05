import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(low=0, high=1, size=input_size)
        self.bias = np.random.uniform(low=0, high=1)
        self.last_weighted_sum = 0

    def modify_w(self, w_values):
        self.weights = w_values

    def get_w(self):
        return self.weights

    def update_w(self, dw_values):
        self.weights += dw_values


    def get_weighted_sum(self, data, bias:0):
        self.last_weighted_sum = np.sum(self.weights * data) + bias
        return self.last_weighted_sum
    
    def activate(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
