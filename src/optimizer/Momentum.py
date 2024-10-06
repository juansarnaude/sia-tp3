from src.optimizer.Optimizer import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def initialize_velocity(self, layer):
        if layer not in self.velocities:
            self.velocities[layer] = {
                'weights': np.zeros_like(layer.get_weights()), 
                'biases': np.zeros_like(layer.get_biases())
            }

    def update(self, layer, gradients):
        self.initialize_velocity(layer)
        velocity = self.velocities[layer]
        
        for j, neuron in enumerate(layer.neurons):
            grad_w, grad_b = gradients[j]
            velocity['weights'][j] = self.momentum * velocity['weights'][j] + self.learning_rate * grad_w
            velocity['biases'][j] = self.momentum * velocity['biases'][j] + self.learning_rate * grad_b
            neuron.weights += velocity['weights'][j]
            neuron.bias += velocity['biases'][j]
