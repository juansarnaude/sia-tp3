from tensorflow import keras
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.utils.functions import tanh, sigmoid
from src.optimizer.GradientDescent import GradientDescent

import numpy as np

def index_of_max_value(float_list):
    if not float_list:
        raise ValueError("The list is empty")
    return float_list.index(max(float_list))


def save_model(model, filename):
    data = {
        "weights": [layer.get_weights() for layer in model.layers],
        "biases": [layer.get_biases() for layer in model.layers]
    }
    np.savez(filename, **data)

def load_model(model, filename):
    data = np.load(filename)
    for i, layer in enumerate(model.layers):
        for j, neuron in enumerate(layer.neurons):
            neuron.weights = data['weights'][i][j]
            neuron.bias = data['biases'][i][j]

# # Example usage:
# save_model(mlp_model, "model.npz")

# # Example usage:
# load_model(mlp_model, "model.npz")

### Load the dataset (automatically downloads if not already present)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x are the images and y are the labels
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Example of how to access the data
print(f'Training data: {x_train.shape} images, {y_train.shape} labels')
print(f'Test data: {x_test.shape} images, {y_test.shape} labels')

### Prepare the data

# Training data
flattened_images = x_train.reshape(60000, 28 * 28)
inputs = [np.array(image) for image in flattened_images]
# inputs = flattened_images.tolist()
# print(f"shape of inputs: {type(inputs[0])} {len(inputs[0])}")
raw_expected_values = y_train.tolist()
expected_values = [[1 if i == number else 0 for i in range(10)] for number in raw_expected_values]

# Testing data
test_flattened_images = x_test.reshape(10000, 28 * 28)
test_inputs = [np.array(image) for image in test_flattened_images]
# test_inputs = flattened_images.tolist()

raw_test_expected_values = y_test.tolist()
test_expected_values = [[1 if i == number else 0 for i in range(10)] for number in raw_test_expected_values]


# print(f"len expected values: {len(expected_values)}")

### Set up the multi layer perceptron

# Input layer has 28x28 = 784 neurons for input
layer_sizes = [784, 256, 128, 10]
learning_rate = 0.01
activation_function = tanh
optimizer = GradientDescent(learning_rate)
epochs = 10
epsilon = 0.00001

mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_function=activation_function,
        optimizer=optimizer
    )

mlp.train(inputs, expected_values, epochs=epochs, epsilon=epsilon)

print("saving the model")
save_model(mlp, "model.npz")

positives = 0
for i, input in enumerate(test_inputs):
    prediction = mlp.predict(input)
    if index_of_max_value(prediction) == raw_test_expected_values[i]:
        positives = positives + 1
print(f"La cantidad de aciertos es: {positives}/10000")






