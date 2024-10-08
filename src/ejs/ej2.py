import json
import pandas as pd
import numpy as np
import random
from src.perceptrons.PerceptronLinear import PerceptronLinear

if __name__ == "__main__":
    with open("./configs/ej2.json") as file:
        config = json.load(file)

    df = pd.read_csv(config["input_file"])

    perceptron = PerceptronLinear(len(df.iloc[0]) - 1, config["learning_rate"], )

    k = config["k"]
    # list of inputs
    inputs = np.array(df[['x1','x2','x3']].values.tolist())

    # list of expected values
    expected_values = np.array(df[['y']].values.tolist())

    k_fold = int(len(inputs) / k)



    sectors_inputs = []
    sectors_expected_values = []
    for i in range(k):
        print(i)
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
        if(i == len(sectors_inputs) - 1):
            training_inputs = sectors_inputs[:i]
            training_expected_values = sectors_expected_values[:i]
        else:
            training_inputs = sectors_inputs[:i] + sectors_inputs[(i+1):]
            training_expected_values = sectors_expected_values[:i] + sectors_expected_values[(i+1):]
        #print("testing_inputs",testing_inputs)
        #print("testing_expected_values",testing_expected_values)
        #print("training_inputs",training_inputs)
        #print("training_expected_values",training_expected_values)
        #llamar a evaluate
        #appendear a un array de metrics



    perceptron.run(config["periods"], config["epsilon"], df, config["output_file"])

