import numpy as np
import nn_from_scratch.functions as F

class Layer:
    """
    Represents a single, fully-connected layer in a neural net.
    Handles forward and backward passes, parameter updates, counts and RAM usage.
    Attributes:
        input_size (int): Number of inputs to the layer.
        output_size (int): Number of outputs from the layer.
        activation_function (callable): Activation function applied to the layer's output.
        random_seed (int): Seed for code reproducibility.
    """
    def __init__(self, input_size: int, output_size: int, activation_function: str = None, random_seed: int = None):

        self.input_size = input_size
        self.output_size = output_size

        # Initialize possibly small random weights and biases using normal distribution (avoid vanishing gradients)
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
        self.biases = np.zeros((1, output_size), dtype=np.float32)

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
        """Sets the activation function for the layer."""
        if activation_function not in F.activation_functions:
            raise ValueError(f"Sorry, the activation function '{activation_function}' is not yet supported.")
        self.activation_function = F.activation_functions[activation_function]

    def reset_weights(self):
        """Resets the weights and biases of the layer."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = np.random.randn(self.input_size, self.output_size).astype(np.float32) * 0.01
        self.biases = np.zeros((1, self.output_size), dtype=np.float32)

    # Forward pass through the layer
    def forward(self, inputs):
        """Performs the forward pass through the layer."""
        self.inputs = inputs
        self.weighted_sum = np.dot(self.inputs, self.weights) + self.biases
        self.outputs = self.activation_function(self.weighted_sum, derivative=False).astype(np.float32)
        return self.outputs

    # Backward pass through the layer
    def backward(self, d_previous):
        """Performs the backward pass through the layer."""
        try:
            # Compute the gradient of the next layer/loss w.r.t. the weighted sum (dL/dZ)
            d_weighted_sum = d_previous * self.activation_function(
                self.weighted_sum, derivative=True
                )
        except NotImplementedError: # When using softmax with cross-entropy loss
            d_weighted_sum = d_previous

        # Gradients for weights and biases
        d_weights = np.dot(self.inputs.T, d_weighted_sum).astype(np.float32)
        d_biases = np.sum(d_weighted_sum, axis=0, keepdims=True).astype(np.float32)
        # Gradient to be passed to the previous layer
        d_current = np.dot(d_weighted_sum, self.weights.T).astype(np.float32)
        return d_weights, d_biases, d_current

    def update_params(self, d_weights, d_biases, learning_rate: float):
        """Updates the weights and biases after backward pass."""
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
    # Function to get the number of parameters in the layer
    def get_num_params(self):
        """Returns the total number of parameters in the layer."""
        params_count = self.weights.size + self.biases.size
        return params_count

    # Function to get the RAM usage of the layer
    def get_ram_usage(self):
        """
        Returns the total RAM used by the layer's weights and biases.
        During training, activation outputs and gradients will be added to this in the NN class.
        """
        weights_ram = self.weights.nbytes
        biases_ram = self.biases.nbytes
        total_ram_used = weights_ram + biases_ram
        return total_ram_used