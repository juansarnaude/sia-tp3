import json
import pandas as pd
from src.perceptrons.PerceptronNonLinear import PerceptronNonLinear

if __name__ == "__main__":
    with open("./configs/ej2NonLinear.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["beta"], config["activation"])
    perceptron.run(config["periods"], config["epsilon"], df, config["output_file"][:-4] + "_" + config["activation"] + ".csv")