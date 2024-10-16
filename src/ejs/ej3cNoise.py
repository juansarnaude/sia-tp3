from src.metrics.Evaluator import Evaluator
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.optimizer.GradientDescent import GradientDescent
from src.optimizer.Momentum import Momentum
from src.optimizer.Adam import Adam
from src.utils.functions import sigmoid, tanh, gaussian_noise, index_of_max_value, confusion_metrics

import json
import pandas as pd
import numpy as np
import csv

np.random.seed(34567)

dataset_multiplicator = 100

with open("./configs/ej3c.json") as file:
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

    # Create model
    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_function=activation_funciton,
        optimizer=optimizer
    )

    expected_values = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]

    expected_values_10 = []
    for i in range(dataset_multiplicator):
        for expected_value in expected_values:
            expected_values_10.append(expected_value)

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)

        mlp.train(inputs, expected_values, epochs, epsilon)                  

        x = 0
        standard_deviations = []
        while x <= 5:
            standard_deviations.append(x)
            x += 0.25
        
        writer.writerow(["standard_deviation", "f1_score"])  # Encabezados
        
        for standard_deviation in standard_deviations:
            noisy_matrix_list = []
            for i in range(dataset_multiplicator):
                for matrix in matrix_list:
                    noisy_matrix_list.append(gaussian_noise(matrix=matrix, standard_deviation=standard_deviation).values.flatten())

            test_confusion_matrix = np.zeros((10, 10))  
            for test_input, expected_value in zip(noisy_matrix_list, expected_values_10):
                #testing_set
                expected = index_of_max_value(expected_value)
                test_prediction=mlp.predict(test_input)
                test_prediction_normalized = index_of_max_value(test_prediction)
                test_confusion_matrix[expected][test_prediction_normalized] += 1
            
            test_metrics = confusion_metrics(test_confusion_matrix)

            writer.writerow([standard_deviation, test_metrics["macro_f1_score"]])

        