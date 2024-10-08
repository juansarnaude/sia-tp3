from tensorflow import keras
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.utils.functions import tanh, sigmoid
from src.optimizer.GradientDescent import GradientDescent
from src.optimizer.Momentum import Momentum
from src.optimizer.Adam import Adam

import numpy as np
import json

def index_of_max_value(float_list):
    list = float_list.tolist()
    if not list:
        raise ValueError("The list is empty")
    return list.index(max(list))


def save_model(model, filename):
    data = {
        "weights": [layer.get_weights() for layer in model.layers],
        "biases": [layer.get_biases() for layer in model.layers]
    }
    np.savez(filename, **data)

def load_model(model, filename):
    data = np.load(filename, allow_pickle=True)  # Allow pickle to load object arrays
    for i, layer in enumerate(model.layers):
        for j, neuron in enumerate(layer.neurons):
            neuron.weights = data['weights'][i][j]
            neuron.bias = data['biases'][i][j]

# # Example usage:
# save_model(mlp_model, "model.npz")

# # Example usage:
# load_model(mlp_model, "model.npz")

# Input layer has 28x28 = 784 neurons for input


with open("./configs/ej4.json") as file:
    config = json.load(file)

    learning_rate=config["learning_rate"]

    # layers per output
    layer_sizes=config["layer_sizes"]

    # activation_function
    activation_function_str = config["activation_function"]
    if activation_function_str == "tanh":
        activation_function = tanh
    elif activation_function_str == "sigmoid":
        activation_function = sigmoid
    else:
        raise ValueError("invalid activation function argument")

    # Optimizer configuration
    optimizer_config = config["optimizer"]

    optimizer_str = optimizer_config["method"]
    momentum = optimizer_config["momentum"]

    adam_config = optimizer_config["adam"]
    beta_1 = adam_config["beta_1"]
    beta_2 = adam_config["beta_2"]

    optimizer = None
    if optimizer_str == "gradient_descent":
        optimizer = GradientDescent(learning_rate)
    elif optimizer_str == "momentum":
        optimizer = Momentum(learning_rate, momentum)
    elif optimizer_str == "adam":
        optimizer = Adam(learning_rate, beta_1, beta_2)
    else:
        raise ValueError("invalid optimizer method argument")

    # cutoffs
    epochs = config["epochs"]
    epsilon = config["epsilon"]

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






