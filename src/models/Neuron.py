import numpy as np

class Neuron:

    def __init__(self, w_amount):
        self.w = np.zeros(w_amount)
        # TODO load the theta and compute activation from args

    def modify_w(self, w_values):
        self.w = w_values

    def get_w(self):
        return self.w

    def get_weighted_sum(self, data, bias:0):
        return np.sum(self.w * data) + bias

    # TODO load the theta and compute activation from args
    def theta(self, weighted_sum):
        if weighted_sum >= 0:
            return 1
        return -1

    # TODO load the theta and compute activation from args

    def compute_activation(self, theta_value):
        if theta_value > 0:
            return 1
        return -1