from src.optimizer.Optimizer import Optimizer

class GradientDescent(Optimizer):
    def update(self, layer, gradients):
        for j, neuron in enumerate(layer.neurons):
            grad_w, grad_b = gradients[j]
            neuron.weights += self.learning_rate * grad_w
            neuron.bias += self.learning_rate * grad_b
