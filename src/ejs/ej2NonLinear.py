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
    # list of inputs
    inputs = np.array(df[['x1','x2','x3']].values.tolist())

    # list of expected values
    expected_values = np.array(df[['y']].values.tolist())


    if k == 1:
        if config["beta"] == -1:
            betas = [0.1,0.25,0.5,1,2.5,5,10,25,50,100]
            for i in betas:
                perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["epsilon"],config["output_file"])
                perceptron.beta = i
                perceptron.out_file = config["output_file"][:-4] + "b" + str(perceptron.beta) + ".csv"
                perceptron.train(inputs, expected_values, config["periods"],config["epsilon"])
            exit()

        else:
            perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["epsilon"], config["output_file"])
            perceptron.beta = config["beta"]
            perceptron.train(inputs, expected_values, config["periods"], config["epsilon"])
            exit()



    if config["k"] == -1:
        for j in range(10):
            shuffled_df = df.sample(frac=1).reset_index(drop=True)
            inputs = np.array(shuffled_df[['x1', 'x2', 'x3']].values.tolist())
            expected_values = np.array(shuffled_df[['y']].values.tolist())

            # SPLIT RATE
            input_train = []
            input_test = []
            expected_train = []
            expected_test = []

            for i in range(len(expected_values)):
                if i < config["split"]:
                    input_train.append(inputs[i])
                    expected_train.append(expected_values[i][0])
                else:
                    input_test.append(inputs[i])
                    expected_test.append(expected_values[i][0])

                perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["epsilon"],config["output_file"]+str(j)+".csv")
                perceptron.train_and_test(input_train, expected_train, input_test, expected_test, config["periods"], 1)

            exit(1)

    # Preparation of folds
    fold_size = int(len(inputs) / k)
    fold_inputs = []
    fold_expected_values = []
    for i in range(k):
        fold_input_array = []
        fold_expected_values_array = []

        for j in range(fold_size):
            fold_input_array.append(inputs[i * fold_size + j])
            fold_expected_values_array.append(expected_values[i * fold_size + j][0])

        fold_inputs.append(fold_input_array)
        fold_expected_values.append(fold_expected_values_array)


    # K-Runs
    for i in range(k):
        perceptron = PerceptronNonLinear(len(df.iloc[0]) - 1, config["learning_rate"], config["epsilon"], config["output_file"])
        perceptron.beta = config["beta"]

        fold_k_input = []
        fold_k_input_expected = []

        for j in range(k):
            if j != i:
                for x in range(fold_size):
                    fold_k_input.append(fold_inputs[j][x])
                    fold_k_input_expected.append(fold_expected_values[j][x])


        perceptron.train_and_test(fold_k_input, fold_k_input_expected, fold_inputs[i], fold_expected_values[i], config["periods"], i)
        print(fold_expected_values[i])
