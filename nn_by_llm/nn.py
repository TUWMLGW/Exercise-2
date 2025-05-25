import numpy as np

# Activation functions and derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

# Dictionary for activation functions
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}

# Initialize weights and biases
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters

# Forward propagation
def forward_propagation(X, parameters, activations):
    A = X
    cache = {"A0": A}
    L = len(activations)
    for l in range(1, L + 1):
        Z = parameters[f"W{l}"] @ A + parameters[f"b{l}"]
        activation_func = activation_functions[activations[l - 1]][0]
        A = activation_func(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return A, cache

# Compute loss (binary cross-entropy)
def compute_loss(Y_hat, Y):
    m = Y.shape[1]
    loss = -1/m * np.sum(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8))
    return loss

# Backward propagation
def backward_propagation(X, Y, parameters, cache, activations):
    grads = {}
    m = X.shape[1]
    L = len(activations)
    A_final = cache[f"A{L}"]
    dA = -(np.divide(Y, A_final + 1e-8) - np.divide(1 - Y, 1 - A_final + 1e-8))

    for l in reversed(range(1, L + 1)):
        activation_func_deriv = activation_functions[activations[l - 1]][1]
        dZ = dA * activation_func_deriv(cache[f"Z{l}"])
        A_prev = cache[f"A{l - 1}"]
        grads[f"dW{l}"] = 1/m * dZ @ A_prev.T
        grads[f"db{l}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = parameters[f"W{l}"].T @ dZ
    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

# Training loop
def train(X, Y, layer_dims, activations, num_epochs=1000, learning_rate=0.01, print_loss=False):
    parameters = initialize_parameters(layer_dims)
    for i in range(num_epochs):
        Y_hat, cache = forward_propagation(X, parameters, activations)
        loss = compute_loss(Y_hat, Y)
        grads = backward_propagation(X, Y, parameters, cache, activations)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_loss and i % 100 == 0:
            print(f"Epoch {i}: Loss = {loss:.4f}")
    return parameters

# Prediction
def predict(X, parameters, activations):
    Y_hat, _ = forward_propagation(X, parameters, activations)
    return (Y_hat > 0.5).astype(int)
