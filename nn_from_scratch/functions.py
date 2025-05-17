import numpy as np

# Activation Functions
def sigmoid(x: np.ndarray, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)

def tanh(x: np.ndarray, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)

activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    None: None,
}

# Loss Functions
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape of {y_true.shape} and y_pred shape of {y_pred.shape} must match.")
    if derivative:
        return (y_pred - y_true)#2 / y_true.size * (y_pred - y_true) 
    else:
        return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape of {y_true.shape} and y_pred shape of {y_pred.shape} must match.")
    if derivative:
        return np.sign(y_pred - y_true) / y_true.size
    else:
        return np.mean(np.abs(y_true - y_pred))

loss_functions = {
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    None: None,
}