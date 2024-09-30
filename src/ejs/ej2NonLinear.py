import json
import pandas as pd
from src.perceptrons.PerceptronNonLinear import PerceptronNonLinear

if __name__ == "__main__":
    with open("./configs/ej2NonLinear.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronNonLinear(config["learning_rate"], config["periods"], config["epsilon"], df, config["beta"])
    perceptron.run()