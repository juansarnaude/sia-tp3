from src.models.Neuron import Neuron
import numpy as np
from src.perceptrons.MultiLayerPerceptron import sigmoid, tanh

class Layer:
    def __init__(self, n_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(n_neurons)]
        self.last_outputs = np.zeros(n_neurons)

    def forward(self, inputs, bias=0):
        self.last_outputs = np.array([neuron.get_weighted_sum(inputs, bias) for neuron in self.neurons])
        # Apply tanh activation function to the weighted sums
        return tanh(self.last_outputs)

    def get_weights(self):
        return [neuron.get_w() for neuron in self.neurons]

    def update_weights(self, dW):
        for i, neuron in enumerate(self.neurons):
            neuron.update_w(dW[i])