from src.optimizer.Optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, gradients):
        self.t += 1

        if layer not in self.m:
            self.m[layer] = []
            self.v[layer] = []
            for neuron in layer.neurons:
                self.m[layer].append((np.zeros_like(neuron.weights), np.zeros_like(neuron.bias)))
                self.v[layer].append((np.zeros_like(neuron.weights), np.zeros_like(neuron.bias)))

        for i, ((grad_w, grad_b), neuron) in enumerate(zip(gradients, layer.neurons)):
            m_w, m_b = self.m[layer][i]
            v_w, v_b = self.v[layer][i]

            # Update biased first moment estimate
            m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b

            # Update biased second raw moment estimate
            v_w = self.beta2 * v_w + (1 - self.beta2) * np.square(grad_w)
            v_b = self.beta2 * v_b + (1 - self.beta2) * np.square(grad_b)

            # Compute bias-corrected first moment estimate
            m_w_hat = m_w / (1 - self.beta1**self.t)
            m_b_hat = m_b / (1 - self.beta1**self.t)

            # Compute bias-corrected second raw moment estimate
            v_w_hat = v_w / (1 - self.beta2**self.t)
            v_b_hat = v_b / (1 - self.beta2**self.t)

            # Update parameters
            neuron.weights += self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            neuron.bias += self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            # Store updated moments
            self.m[layer][i] = (m_w, m_b)
            self.v[layer][i] = (v_w, v_b)