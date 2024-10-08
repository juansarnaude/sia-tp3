import numpy as np
import csv
from src.models.Layer import Layer

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, activation_function, optimizer, output_path):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activation_function))
        self.optimizer = optimizer
        self.output_path = output_path

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, x, y, outputs):
        # Calcular el error y el gradiente para la capa de salida
        error = y - outputs[-1]
        delta = error * self.layers[-1].activation_function(outputs[-1], derivative=True)

        # Retropropagar el error a través de las capas
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inputs = outputs[i] if i > 0 else x

            # Actualizar pesos y sesgos
            
            gradients = []
            for j, neuron in enumerate(layer.neurons): # Calculate gradients
                grad_w = delta[j] * inputs
                grad_b = delta[j]
                gradients.append((grad_w, grad_b))

            # Use the optimizer to update the weights and biases
            self.optimizer.update(layer=layer, gradients=gradients)

            # Calcular delta para la capa anterior
            if i > 0:
                delta = np.dot(delta, layer.get_weights()) * self.layers[i-1].activation_function(outputs[i], derivative=True)

    def train(self, X, y, epochs, epsilon):
        errors = []  # Lista para almacenar los errores por época

        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            total_error = 0
            for x, target in zip(X, y):
                outputs = [x]
                # Forward pass
                for layer in self.layers:
                    outputs.append(layer.forward(outputs[-1]))

                # Backward pass
                self.backward(x, target, outputs)

                # Calcular error
                error = self.mse(target, outputs[-1])
                total_error += error

            avg_error = total_error / len(X)  # Error promedio por época
            errors.append(avg_error)  # Agregar el error promedio a la lista
            print(f"Época {epoch + 1}/{epochs}, Error: {avg_error:.6f}")

            if avg_error < epsilon:
                print(f"Convergencia alcanzada en la época {epoch + 1}")
                break

        # Guardar errores en un archivo CSV
        with open(self.output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Error'])  # Encabezados
            for epoch, error in enumerate(errors):
                writer.writerow([epoch + 1, error])

    def predict(self, x):
        return self.forward(x)

    @staticmethod
    def mse(y_true, y_pred):
        return 0.5*np.mean((y_true - y_pred) ** 2)
    




