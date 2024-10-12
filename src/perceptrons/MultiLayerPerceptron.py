import numpy as np
import csv
from src.models.Layer import Layer
from src.metrics.Accuracy import Accuracy
from src.metrics.Precision import Precision
from src.metrics.Recall import Recall
from src.metrics.F1Score import F1Score
from src.utils.functions import index_of_max_value

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

    def train(self, X, test_inputs, y, epochs, epsilon):
        errors = []  # Lista para almacenar los errores por época
        test_errors = []  # Lista para almacenar los errores por epoca
        accuracy = []
        f1 = []
        precision = []
        recall = []
        if type(y[0])==list:
            output_set_len = len(y[0])
        else: 
            output_set_len = 2
        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            total_error = 0
            total_test_error = 0
            for x,test_input, target in zip(X,test_inputs, y):
                outputs = [x]
                outputs_test = [test_input]
                # Forward pass
                for layer in self.layers:
                    outputs.append(layer.forward(outputs[-1]))

                # Foward pass for test
                for layer in self.layers:
                    outputs_test.append(layer.forward(outputs_test[-1]))

                # Backward pass
                self.backward(x, target, outputs)
                
                # Calcular error
                error = self.mse(target, outputs[-1])
                total_error += error

                # Calcular error
                test_error = self.mse(target, outputs_test[-1])
                total_test_error += test_error

            avg_error = total_error / len(X)  # Error promedio por época
            errors.append(avg_error)  # Agregar el error promedio a la lista

            avg_test_error = total_test_error / len(test_inputs)  # Error promedio por época
            test_errors.append(avg_test_error)  # Agregar el error promedio a la lista

            confusion_matrix = np.zeros((output_set_len, output_set_len))

            for input, expected_value in zip(test_inputs, y):
                prediction = index_of_max_value(self.predict(input))
                expected = index_of_max_value(expected_value)
                confusion_matrix[expected][prediction] += 1

            # compute the confusion matrix results
            true_positive=0
            true_negative=0
            false_positive=0
            false_negative=0

            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix[i])):
                    if i == j:
                        true_positive += confusion_matrix[i][j]
                    else:
                        false_negative += confusion_matrix[i][j]
                        true_negative += confusion_matrix[j][j]
                        false_positive += confusion_matrix[j][i]

            print(f"false negative: {false_negative}")
            print(f"false postivie: {false_positive}")
            print(f"true positive {true_positive}")
            print(f"true negative: {true_negative}")

            accuracy.append(Accuracy.get_metric(true_positive,true_negative,false_positive,false_negative))
            f1.append(F1Score.get_metric(true_positive,true_negative,false_positive,false_negative))
            precision.append(Precision.get_metric(true_positive,true_negative,false_positive,false_negative))
            recall.append(Recall.get_metric(true_positive,true_negative,false_positive,false_negative))

            print(f"Época {epoch + 1}/{epochs}, Error: {avg_error:.6f}, Test Error: {avg_test_error:.6f}")
            
            if avg_error < epsilon:
                print(f"Convergencia alcanzada en la época {epoch + 1}")
                break

        # Guardar errores en un archivo CSV
        with open(self.output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Error', 'Test Error', 'Accuracy', 'F1Score', 'Precision', 'Recall'])  # Encabezados
            for epoch in range(len(errors)):
                writer.writerow([epoch + 1, errors[epoch], test_errors[epoch],accuracy[epoch],f1[epoch],precision[epoch],recall[epoch]])

    def predict(self, x):
        return self.forward(x)

    @staticmethod
    def mse(y_true, y_pred):
        return 0.5*np.mean((y_true - y_pred) ** 2)
    




