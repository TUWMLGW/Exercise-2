import numpy as np
from tqdm import tqdm
from typing import List

from nn_from_scratch.layer import Layer
import nn_from_scratch.functions as F
import matplotlib.pyplot as plt

class NN:

    def __init__(self, layers: List[Layer], activation_function: str, learning_rate: float, loss_function: str):
        
        for i in range(1, len(layers)):
            if layers[i].input_size != layers[i - 1].output_size:
                raise ValueError(f"Input size of layer {i} does not match output size of layer {i - 1}.")
        self.layers = layers

        self.learning_rate = learning_rate
        for layer in self.layers:
            if not layer.learning_rate:
                layer.set_learning_rate(learning_rate)

        # Set the activation function
        if activation_function in F.activation_functions:
            self.activation_function = F.activation_functions[activation_function]
            for layer in self.layers:
                if not layer.activation_function:
                    layer.set_activation_function(activation_function)
        else:
            raise ValueError(f"Sorry, the activation function '{activation_function}' is not yet supported.")

        # Set the loss function
        if loss_function in F.loss_functions:
            self.loss_function = F.loss_functions[loss_function]
        else:
            raise ValueError(f"Sorry, the loss function '{loss_function}' is not yet supported.")

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Layer should be an instance of the respective class.")
        if not self.layers:
            self.layers.append(layer)
        else:
            previous_layer_output_size = self.layers[-1].output_size
            if layer.input_size != previous_layer_output_size:
                raise ValueError("Input size must match output size of previous layer.")
            self.layers.append(layer)

    # Functions to manage the layer's properties
    def set_activation_function(self, activation_function):
        self.activation_function = activation_function
        for layer in self.layers:
            layer.set_activation_function(activation_function)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        outputs = inputs
        return outputs

    def train_step(self, inputs, targets):
        outputs = self.predict(inputs)
        error = self.loss_function(targets, outputs)
        de = self.loss_function(targets, outputs, derivative=True)
        for layer in self.layers[::-1]:
            de = layer.backward(de)
        return error

    def train(self, inputs, targets, epochs, verbose=False, visualize=False):

        if self.layers[0].input_size != inputs.shape[1]:
            raise ValueError(f"Input size of {inputs.shape[1]} does not match the input size of the first layer {self.layers[0].input_size}.")
        if self.layers[-1].output_size != targets.shape[1]:
            raise ValueError(f"Output size of {targets.shape[1]} does not match the output size of the last layer {self.layers[-1].output_size}.")


        errors = []
        for epoch in tqdm(range(epochs), desc="Training"):
            current_errors = []
            for i in range(len(inputs)):
                current_error = self.train_step(inputs[i].reshape(1, -1), targets[i].reshape(1, -1))
                current_errors.append(current_error)
            errors.append(np.mean(current_errors))
            if verbose:
                print(f"Epoch {epoch + 1} out of {epochs}: Error: {np.mean(current_errors):.4f}")
        if visualize:
            plt.plot(errors)
            plt.xlabel("Epoch")
            plt.ylabel("Error")