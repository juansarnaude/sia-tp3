import json
import pandas as pd

from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron

if __name__ == "__main__":
    with open("./configs/ej3.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    multi_layer_perceptron = MultiLayerPerceptron( config["neurons_per_layer"], config["learning_rate"], len(df.iloc[0]) - 1)
    multi_layer_perceptron.run(df, config["periods"], config["epsilon"])
