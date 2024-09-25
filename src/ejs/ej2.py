import json
import pandas as pd
from src.perceptrons.PerceptronLinear import PerceptronLinear

if __name__ == "__main__":
    with open("./configs/ej2.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronLinear(config["learning_rate"], config["periods"], config["epsilon"], df)
    perceptron.run()