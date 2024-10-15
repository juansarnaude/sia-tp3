import csv
import json
import pandas as pd
import numpy as np
from src.utils.functions import sigmoid, tanh, gaussian_noise, index_of_max_value, confusion_metrics
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

    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_function=activation_funciton,
        optimizer=optimizer
    )

    #mlp.train(flattened_matrixes, expected_values, epochs=epochs, epsilon=epsilon)

    if config["metrics"]:
        with open(output_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'accuracy', 'precision', 'recall', 'f1_score','mse'])  # Encabezados
            epochs_per_iteration = 1
            iterations = int(epochs/epochs_per_iteration)

            for i in range(iterations):

                if mlp.train(inputs, expected_values, epochs_per_iteration, epsilon):
                    break

                confusion_matrix  = np.zeros((2, 2))
                total_train_error=0

                for train_input, expected_value in zip(inputs, expected_values):

                    prediction = mlp.predict(train_input)
                    total_train_error += mlp.mse(expected_value, prediction)
                    prediction_normalized = index_of_max_value(prediction)

                    confusion_matrix[expected_value][prediction_normalized] += 1

                train_error = total_train_error/len(inputs)
                train_metrics = confusion_metrics(confusion_matrix)

                writer.writerow([epochs_per_iteration * i, train_metrics["accuracy"], train_metrics["macro_precision"], train_metrics["macro_recall"], train_metrics["macro_f1_score"],train_error])






    # COSAS A TESTEAR

    '''
    1) ARQUITECTURA
    2) FUNCION DE ACTIVACION
    3) METODO DE OPTIMIZACION
    4) LEARNING RATE
    '''



    # Add noise to the matrix
    # standard_deviation = config["gaussian_noise"]
    # noisy_input = []
    # for matrix in matrix_list:
    #     noisy_input.append(gaussian_noise(matrix=matrix, standard_deviation=standard_deviation).values.flatten())
    #
    # training_input = []
    # for matrix in matrix_list:
    #     training_input.append(gaussian_noise(matrix=matrix, standard_deviation=0).values.flatten())
    #
    # mlp = MultiLayerPerceptron(
    #     layer_sizes=layer_sizes,
    #     activation_function=activation_funciton,
    #     optimizer=optimizer,
    #     output_path=output_path
    # )
    #
    # mlp.train(training_input, noisy_input, expected_values, epochs=epochs, epsilon=epsilon)
    #
    # for input in inputs:
    #     prediction = mlp.predict(input)
    #     print(f"Entrada: {input}, Predicción: {prediction[0]:.4f}")