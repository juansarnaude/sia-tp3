from src.models.Neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.activation_function = activation_function

    def forward(self, inputs):
        outputs = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.activation_function(outputs)

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])