import json
import pandas as pd
import numpy as np
import random
from src.perceptrons.PerceptronNonLinear import PerceptronNonLinear


def normalize_to_range(data, new_min, new_max):
    """Normalizes array to a new range given by [new_min - new_max]"""
    min_val = min(data)
    max_val = max(data)
    aux = np.concatenate((min_val, max_val))
    normalized_data = np.interp(data, aux, (new_min, new_max))
    return normalized_data


if __name__ == "__main__":
    with open("./configs/ej2NonLinear.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"],config["epsilon"],config["output_file"] )



    k = config["k"]
    # list of inputs
    inputs = np.array(df[['x1','x2','x3']].values.tolist())

    # list of expected values
    expected_values = np.array(df[['y']].values.tolist())


    if k == 1 and config["beta"] == -1:
        for i in range(1,10):
            perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["epsilon"],config["output_file"])
            perceptron.beta = i/2
            perceptron.out_file = config["output_file"][:-4] + "b" + str(perceptron.beta) + ".csv"
            perceptron.train(inputs, expected_values, config["periods"],config["epsilon"])
        exit()


    expected_values = normalize_to_range(expected_values, -1, 1).tolist()
    perceptron.train(inputs, expected_values,config["periods"],config["epsilon"])

    k_fold = int(len(inputs) / k)


    sectors_inputs = []
    sectors_expected_values = []
    for i in range(k):
        aux = []
        aux2 = []

        for j in range(k_fold):
            random_number = random.randint(0,len(inputs)-1)
            aux.append(inputs[random_number])
            inputs = np.delete(inputs,random_number,axis=0)
            aux2.append(expected_values[random_number])
            expected_values = np.delete(expected_values,random_number,axis=0)
        sectors_inputs.append(aux)
        sectors_expected_values.append(aux2)


    for i in range(len(sectors_inputs)):
        testing_inputs = sectors_inputs[i]
        testing_expected_values = sectors_expected_values[i]
        if i == len(sectors_inputs) - 1:
            training_inputs = np.concatenate(sectors_inputs[:i])
            training_expected_values = np.concatenate(sectors_expected_values[:i])
        elif i == 0:
            training_inputs = np.concatenate(sectors_inputs[(i + 1):])
            training_expected_values = np.concatenate(sectors_expected_values[(i + 1):])
        else:
            training_inputs = np.concatenate(sectors_inputs[:i] + sectors_inputs[(i + 1):])
            training_expected_values = np.concatenate(sectors_expected_values[:i] + sectors_expected_values[(i + 1):])
        perceptron.train_and_test(training_inputs, training_expected_values,testing_inputs,testing_expected_values, config["periods"],i)






