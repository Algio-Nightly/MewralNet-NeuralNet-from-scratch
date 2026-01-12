import numpy as np
def modified_tanh(array:np.ndarray):
    return np.tanh(array)

def sigmoid(array:np.ndarray):
    return 1/(1+np.exp(-array))

def modified_sigmoid(array:np.ndarray) -> np.ndarray:
    array = 1 / (1 + np.exp(-array))
    array = array*4 - 2
    return array

def sigmoid_derivative(array:np.ndarray):
    return array * (1.0 - array)

def tanh_derivative(array:np.ndarray):
    return 1.0 - (array ** 2) 

def modified_sigmoid_derivative(array:np.ndarray):
    return 1.0 - (array ** 2)/4.0
