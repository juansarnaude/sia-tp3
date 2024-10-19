import json
import pandas as pd
import numpy as np
import random
from src.perceptrons.PerceptronNonLinear import PerceptronNonLinear



if __name__ == "__main__":
    with open("./configs/ej2NonLinear.json") as file:
        config = json.load(file)

        df = pd.read_csv(config["input_file"])

        k = config["k"]

        input_size = 3
        learning_rate = config["learning_rate"]
        epsilon = config["epsilon"]
        epochs = config["epochs"]

        betas = [0.1,0.25,0.5,1,2.5,5,10,25,50,100]

        # list of inputs
        inputs = np.array(df[['x1','x2','x3']].values.tolist())

        # list of expected values
        expected_values = np.array(df[['y']].values.tolist())
        expected_values = [expected_values[i][0] for i in range(len(expected_values))]

        for beta in betas:
            nlp = PerceptronNonLinear(input_size=3, learning_rate=learning_rate, epsilon=epsilon, beta=beta)
            nlp.fit(inputs, expected_values, epochs=epochs, shuffle=False)
            predictions = []
            for input in inputs:
                predictions.append(nlp.predict(input))
            
            print(nlp.error(predictions, expected_values))

