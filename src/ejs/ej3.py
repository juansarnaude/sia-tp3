import json
import pandas as pd
import numpy as np
from src.utils.functions import sigmoid, tanh
from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron

if __name__ == "__main__":
    with open("./configs/ej3.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])
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
    
    epochs = config["epochs"]
    epsilon = config["epsilon"]

    # list of inputs
    inputs = np.array(df[['x1', 'x2']].values.tolist())

    # list of expected values
    expected_values = np.array(df[['y']].values.tolist())

    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_function=activation_funciton,
        learning_rate=learning_rate
    )

    mlp.train(inputs, expected_values, epochs=epochs, epsilon=epsilon)

    for input in inputs:
        prediction = mlp.predict(input)
        print(f"Entrada: {input}, Predicción: {prediction[0]:.4f}")