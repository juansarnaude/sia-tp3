import json
import pandas as pd
from src.perceptrons.PerceptronStep import PerceptronStep

if __name__ == "__main__":
    with open("./configs/ej1.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronStep(len(df.iloc[0]) - 1, config["learning_rate"],)
    perceptron.run(config["periods"], config["epsilon"], df)