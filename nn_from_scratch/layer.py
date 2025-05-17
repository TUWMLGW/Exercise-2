import numpy as np
import nn_from_scratch.functions as F
class Layer:
    
    def __init__(self, input_size: int, output_size: int, activation_function: str = None, learning_rate: float = None):

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Set the activation function
        if activation_function in F.activation_functions:
            self.activation_function = F.activation_functions[activation_function]
        else:
            raise ValueError(f"Sorry, the activation function '{activation_function}' is not yet supported.")

        # Initialize random weights and biases
        np.random.seed(0)
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    # Functions to manage the layer's properties
    def set_activation_function(self, activation_function):
        self.activation_function = F.activation_functions[activation_function]

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def reset_weights(self):
        self.weights = np.random.randn(self.input_size, self.output_size)
        self.biases = np.zeros((1, self.output_size))

    # Forward pass through the layer
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z)
        return self.a

    # Backward pass through the layer
    def backward(self, output_gradient):
        activation_derivative = self.activation_function(self.z, derivative=True)
        weight_gradient = self.inputs.reshape(-1, 1).dot((output_gradient * activation_derivative).reshape(1, -1))
        bias_gradient = np.sum(output_gradient * activation_derivative, axis=0, keepdims=True)

        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

        input_gradient = (output_gradient * activation_derivative).dot(self.weights.T)
        
        return input_gradient