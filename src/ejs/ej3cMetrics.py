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
    
    testing_expected_values = [0,1,2,3,4,5,6,7,8,9]

    # Create noisy matrix
    standard_deviation = config["gaussian_noise"]
    noisy_matrix_list = []
    for matrix in matrix_list:
        noisy_matrix_list.append(gaussian_noise(matrix=matrix, standard_deviation=standard_deviation).values.flatten())

    # Data set generator

    test_inputs = noisy_matrix_list

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'accuracy', 'precision', 'recall', 'f1_score', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score'])  # Encabezados
        epochs_per_iteration = 1
        iterations = int(epochs/epochs_per_iteration)
        for i in range(iterations):
            mlp.train(inputs, expected_values, epochs_per_iteration, epsilon)

            # Evaluate training set

            confusion_matrix  = np.zeros((10, 10))
            test_confusion_matrix  = np.zeros((10, 10))

            for train_input, test_input, expected_value in zip(inputs, test_inputs, expected_values):
                prediction = index_of_max_value(mlp.predict(train_input))
                test_prediction = index_of_max_value(mlp.predict(test_input))
                expected = index_of_max_value(expected_value)
                confusion_matrix[expected][prediction] += 1
                test_confusion_matrix[expected][test_prediction] += 1
            
            train_metrics = confusion_metrics(confusion_matrix)
            test_metrics = confusion_metrics(test_confusion_matrix)
            writer.writerow([i*epochs_per_iteration, train_metrics["accuracy"], train_metrics["macro_precision"], train_metrics["macro_recall"], train_metrics["macro_f1_score"], test_metrics["accuracy"], test_metrics["macro_precision"], test_metrics["macro_recall"], test_metrics["macro_f1_score"]])

            