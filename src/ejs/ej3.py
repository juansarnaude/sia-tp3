import json
import pandas as pd
import numpy as np

from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron

if __name__ == "__main__":
    # with open("./configs/ej3.json") as file:
    #     config = json.load(file)

    # df = pd.read_csv(config["input_file"])

    # multi_layer_perceptron = MultiLayerPerceptron( config["neurons_per_layer"], config["learning_rate"], len(df.iloc[0]) - 1)
    # multi_layer_perceptron.run(df, config["periods"], config["epsilon"])
    # Initialize MLP with 3 layers: input size 3, hidden layer with 4 neurons, output layer with 1 neuron

    ###############################

    mlp = MultiLayerPerceptron([2, 4, 1])

    # XOR problem dataset
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Train the network on the XOR problem
    mlp.train(X_train, y_train, learning_rate=0.1, epochs=10000)

    # Test the trained MLP on XOR inputs
    for X in X_train:
        print(f"Input: {X}, Predicted Output: {mlp.predict(X)}")