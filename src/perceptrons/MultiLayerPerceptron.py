import numpy as np

from src.utils.functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative

from src.models.Layer import Layer

class MultiLayerPerceptron:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1]))

    def forward(self, X, bias=0):
        # Forward pass through the layers
        for layer in self.layers:
            X = layer.forward(X, bias)
        return X

    def backpropagate(self, X, y, learning_rate, bias=0):
        # Forward pass to get outputs
        outputs = [X]
        for layer in self.layers:
            X = layer.forward(X, bias)
            outputs.append(X)

        # Calculate delta for the output layers
        deltas = [None] * len(self.layers)
        last_output = outputs[-1]
        deltas[-1] = (y - last_output) * tanh_derivative(self.layers[-1].last_outputs)

        # Backpropagate through layers
        for l in reversed(range(len(self.layers) - 1)):
            deltas[l] = (np.dot(deltas[l + 1], np.array(self.layers[l + 1].get_weights())) * tanh_derivative(self.layers[l].last_outputs))

        # Update weights for each layer
        for l, layer in enumerate(self.layers):
            inputs = outputs[l]  # Inputs to the current layer
            dW = learning_rate * np.outer(deltas[l], inputs)
            layer.update_weights(dW)

    def train(self, X_train, y_train, learning_rate, bias=0, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for X, y in zip(X_train, y_train):
                self.backpropagate(X, y, learning_rate, bias)
                total_loss += np.mean((self.forward(X) - y) ** 2)  # Mean Squared Error
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(X_train)}")

    def predict(self, X):
        return self.forward(X)

