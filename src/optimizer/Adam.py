from src.optimizer.Optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.moments = {}
        self.timestep = 1

    def initialize_moments(self, layer):
        if layer not in self.moments:
            self.moments[layer] = {
                'm_w': np.zeros_like(layer.get_weights()), 
                'v_w': np.zeros_like(layer.get_weights()),
                'm_b': np.zeros_like(layer.get_biases()), 
                'v_b': np.zeros_like(layer.get_biases())
            }

    def update(self, layer, gradients):
        self.initialize_moments(layer)
        moments = self.moments[layer]

        for j, neuron in enumerate(layer.neurons):
            grad_w, grad_b = gradients[j]

            # Update first moment (mean of gradients)
            moments['m_w'][j] = self.beta_1 * moments['m_w'][j] + (1 - self.beta_1) * grad_w
            moments['m_b'][j] = self.beta_1 * moments['m_b'][j] + (1 - self.beta_1) * grad_b

            # Update second moment (mean of squared gradients)
            moments['v_w'][j] = self.beta_2 * moments['v_w'][j] + (1 - self.beta_2) * (grad_w ** 2)
            moments['v_b'][j] = self.beta_2 * moments['v_b'][j] + (1 - self.beta_2) * (grad_b ** 2)

            # Bias-corrected moments
            m_w_hat = moments['m_w'][j] / (1 - self.beta_1 ** self.timestep)
            v_w_hat = moments['v_w'][j] / (1 - self.beta_2 ** self.timestep)
            m_b_hat = moments['m_b'][j] / (1 - self.beta_1 ** self.timestep)
            v_b_hat = moments['v_b'][j] / (1 - self.beta_2 ** self.timestep)

            # Update weights and biases
            neuron.weights += self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            neuron.bias += self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        self.timestep += 1
