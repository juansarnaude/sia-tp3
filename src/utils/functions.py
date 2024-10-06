import numpy as np

# Activation functions and its derivative

def sigmoid(x, derivative=False):
    t = 1 / (1 + np.exp(-x))
    if derivative:
        return t * (1 - t) #TODO: Agregar un
    return t

def tanh(x, derivative=False):
    t = np.tanh(x)
    if derivative:
        return 1 - t**2
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