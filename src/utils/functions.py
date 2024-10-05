import numpy as np

# Activation function and its derivative
def sigmoid(x, derivative=False):
    t = 1 / (1 + np.exp(-x))
    if derivative:
        return t * (1 - t) #TODO: Agregar un
    return t

# Funciones de activación
def tanh(x, derivative=False):
    t = np.tanh(x)
    if derivative:
        return 1 - t**2
    return t