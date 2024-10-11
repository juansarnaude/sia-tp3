import json
import pandas as pd
import numpy as np
from src.utils.functions import sigmoid, tanh, gaussian_noise
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.optimizer.GradientDescent import GradientDescent
from src.optimizer.Momentum import Momentum
from src.optimizer.Adam import Adam

np.random.seed(345678)  

if __name__ == "__main__":
    with open("./configs/ej3b.json") as file:
        config = json.load(file)

    output_path = config["output_file"]

    df = pd.read_csv(config["input_file"], delimiter=' ', header=None)

    df = df.iloc[:, :-1]

    matrix_list = [df.iloc[i:i + 7, :] for i in range(0, len(df), 7)]
    flattened_matrixes = [matrix.values.flatten() for matrix in matrix_list]

    learning_rate=config["learning_rate"]

    # layers per output
    layer_sizes=config["layer_sizes"]

    # activation_function
    activation_function_str = config["activation_function"]
    if activation_function_str == "tanh":
        activation_funciton = tanh
    elif activation_function_str == "sigmoid":
        activation_funciton = sigmoid
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

    # list of inputs
    inputs = flattened_matrixes

    print(flattened_matrixes)

    # list of expected values
    expected_values = [
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0
    ]

    # Add noise to the matrix
    standard_deviation = config["gaussian_noise"]
    noisy_input = []
    for matrix in matrix_list:
        noisy_input.append(gaussian_noise(matrix=matrix, standard_deviation=standard_deviation).values.flatten())

    training_input = []
    for matrix in matrix_list:
        training_input.append(gaussian_noise(matrix=matrix, standard_deviation=0).values.flatten())

    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_function=activation_funciton,
        optimizer=optimizer,
        output_path=output_path
    )

    mlp.train(training_input, noisy_input, expected_values, epochs=epochs, epsilon=epsilon)

    for input in inputs:
        prediction = mlp.predict(input)
        print(f"Entrada: {input}, Predicción: {prediction[0]:.4f}")