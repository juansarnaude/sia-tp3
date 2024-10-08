import random

import numpy as np


# Activation functions and its derivative

def sigmoid(x, derivative=False, beta=1):
    t = 1 / (1 + np.exp(-2*beta*x))
    if derivative:
        return 2 * beta * t * (1 - t)
    return t

def tanh(x, derivative=False, beta=1):
    t = np.tanh(beta*x)
    if derivative:
        return beta * (1 - t**2)
    return t

# Gaussian noise function

def gaussian_noise(matrix, mean=0, standard_deviation=0.1):
    """
    Aplica ruido gaussiano a una matriz de ceros y unos.
    
    Parámetros:
    matriz (numpy array): Matriz original de ceros y unos (por ejemplo, de 7x5).
    media (float): La media de la distribución normal del ruido.
    desviacion_estandar (float): La desviación estándar de la distribución normal del ruido.
    
    Retorna:
    numpy array: Matriz con ruido gaussiano agregado.
    """
    # Convertimos la matriz a float para que el ruido pueda sumarse correctamente
    matriz_float = matrix.astype(float)
    
    # Generamos ruido gaussiano con la misma forma que la matriz
    ruido = np.random.normal(mean, standard_deviation, matrix.shape)
    
    # Sumamos el ruido a la matriz original
    matriz_con_ruido = matriz_float + ruido
    
    # Podemos forzar los valores a estar entre 0 y 1 si fuera necesario (opcional)
    matriz_con_ruido = np.clip(matriz_con_ruido, 0, 1)
    
    return matriz_con_ruido

def k_fold_cross_validation(mlp,inputs,expected_values,k,metrics,epochs,epsilon):
    if(len(inputs) % k != 0):
        print("La longitud de inputs debe ser divisible por k.")
        raise RuntimeError
    k_fold = len(inputs) / k
    training_inputs = [k_fold]
    training_expected_values = [k_fold]
    for i in range(k):
        aux = []
        aux2 = []
        for j in range(k_fold):
            random_number = random.randint(0,len(inputs)-1)
            aux.append(inputs.pop(random_number))
            aux2.append(expected_values.pop(random_number))
        training_inputs[i].append(aux)
        expected_values[i].append(aux2)
        mlp.train(training_inputs,training_expected_values,epochs,epsilon)
        output = []
        for input in training_inputs:
            output.append(mlp.predict(input))
        true_positive,false_positive,true_negative,false_negative = classify(output,expected_values)
        for metric in metrics:
            metric.get_metric(true_positive,true_negative,false_positive,false_negative)


def classify(output,expected_values):
    true_positive,false_positive,true_negative,false_negative = 0,0,0,0
    for i in range(len(output)):
        if output[i] > 0.5:
            if expected_values[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if expected_values[i] == 0:
                true_negative += 1
            else:
                false_negative += 1

    return true_positive,false_positive,true_negative,false_negative

def index_of_max_value(float_list):
    list = float_list.tolist()
    if not list:
        raise ValueError("The list is empty")
    return list.index(max(list))