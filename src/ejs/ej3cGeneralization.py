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
    training_inputs = flattened_matrixes[4:7]

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
    
    train_expected_values = expected_values[4:7]

    print("training expected values")

    print(train_expected_values)

    print('-------------------------')

    print('flattened training inputs')

    print(training_inputs)

    # Create noisy matrix
    standard_deviation = config["gaussian_noise"]
    noisy_matrix_list = []

    for i in [0, 0.05, 0.1, 0.15, 0.2]:
        for matrix in matrix_list:
            noisy_matrix_list.append(gaussian_noise(matrix=matrix, standard_deviation=i).values.flatten())

    testing_expected_values = []
    for i in range(5):
        for expected_value in expected_values:
            testing_expected_values.append(expected_value)

    # Data set generator

    test_inputs = noisy_matrix_list

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"{i}" for i in range(10)])
        epochs_per_iteration = 1000
        iterations = int(epochs/epochs_per_iteration)
        for i in range(iterations):
            if mlp.train(training_inputs, train_expected_values, epochs_per_iteration, epsilon):
                break

        # Evaluate training set
        test_confusion_matrix  = np.zeros((10, 10))

        total_train_error=0
        total_test_error=0

        for test_input, expected_value in zip(test_inputs, testing_expected_values):
            expected = index_of_max_value(expected_value)

            #testing_set
            test_prediction=mlp.predict(test_input)
            total_test_error+=mlp.mse(expected_value, test_prediction)
            test_prediction_normalized = index_of_max_value(test_prediction)
            test_confusion_matrix[expected][test_prediction_normalized] += 1

        print(test_confusion_matrix)

        #testing_set
        test_error=total_test_error/len(test_input)
        test_metrics = confusion_metrics(test_confusion_matrix)

        print(test_metrics["recall_per_class"])
        writer.writerow(test_metrics["recall_per_class"])

            