import numpy as np

class Neuron:
    def __init__(self, w_amount):
        self.w = np.zeros(w_amount)

    def modify_w(self, w_values):
        self.w = w_values

    def get_w(self):
        return self.w

    def update_w(self, dw_values):
        self.w += dw_values


    def get_weighted_sum(self, data, bias:0):
        return np.sum(self.w * data) + bias
