import numpy as np
import nn.functions as F

class Layer:
    
    def __init__(self, input_size: int, output_size: int, activation_function: str = None, random_seed: int = None):

        self.input_size = input_size
        self.output_size = output_size

        # Initialize random weights and biases
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(1, output_size)

        # Set the activation function
        if activation_function in F.activation_functions:
            self.activation_function = F.activation_functions[activation_function]
        else:
            raise ValueError(f"Sorry, the activation function '{activation_function}' is not yet supported.")

        # Variables to store intermediate values for backprop (which are going to be set during forward)
        self.inputs = None  # Stores the input to this layer
        self.weighted_sum = None # Stores Z = XW + B before activation
        self.outputs = None  # Stores A = activation(Z) - i.e. layer output

    # Functions to manage the layer's properties
    def set_activation_function(self, activation_function):
        self.activation_function = F.activation_functions[activation_function]

    def reset_weights(self):
        self.weights = np.random.rand(self.input_size, self.output_size)
        self.biases = np.random.rand((1, self.output_size))

    # Forward pass through the layer
    def forward(self, inputs):
       self.inputs = inputs
       self.weighted_sum = np.dot(self.inputs, self.weights) + self.biases
       self.outputs = self.activation_function(self.weighted_sum, derivative=False)
       return self.outputs

    # Backward pass through the layer
    def backward(self, d_previous):
        try:
            d_weighted_sum = d_previous * self.activation_function(self.weighted_sum, derivative=True) # dL/dZ
        except NotImplementedError: # When using softmax
            d_weighted_sum = d_previous

        d_weights = np.dot(self.inputs.T, d_weighted_sum) # dL/dw
        d_biases = np.sum(d_weighted_sum, axis=0, keepdims=True) # dL/dB
        d_current = np.dot(d_weighted_sum, self.weights.T) # dL/d
        return d_weights, d_biases, d_current

    def update_params(self, d_weights, d_biases, learning_rate: float):
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        