import numpy as np
from tqdm import tqdm
from typing import List

from nn_from_scratch.layer import Layer
import nn_from_scratch.functions as F
import matplotlib.pyplot as plt

class NN:

    def __init__(self, layers: List[Layer], num_classes, activation_function: str, loss_function: str):
        
        for i in range(1, len(layers)):
            if layers[i].input_size != layers[i - 1].output_size:
                raise ValueError(f"Input size of layer {i} does not match output size of layer {i - 1}.")
            if layers[-1].output_size != num_classes:
                raise ValueError(f"Output size of the last layer must correspond to the number of classes to predict")
        self.layers = layers
        self.num_classes = num_classes

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

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, inputs, targets, epochs, learning_rate, batch_size=32, verbose=False, visualize=False):
        n = inputs.shape[0]
        num_batches = int(np.ceil(n / batch_size))
        losses = []
        for epoch in tqdm(range(epochs)):
            permutation = np.random.permutation(n)
            X = inputs[permutation]
            y = targets[permutation]

            epoch_loss = 0
            total_samples = 0
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n)

                X_i = X[start:end]
                y_i = y[start:end]

                if X_i.shape[0] == 0:
                    continue

                prediction = self.predict(X_i)

                loss = self.loss_function(y_i, prediction)
                batch_size_actual = X_i.shape[0]
                epoch_loss += loss * batch_size_actual
                total_samples += batch_size_actual
                d_loss = self.loss_function(y_i, prediction, derivative=True)
                d_previous = d_loss

                for layer in self.layers[::-1]:
                    d_weights, d_biases, d_current = layer.backward(d_previous)
                    layer.update_params(d_weights, d_biases, learning_rate)
                    d_previous = d_current
            
            mean_epoch_loss = epoch_loss / total_samples
            losses.append(mean_epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} ============ Loss: {mean_epoch_loss:.3f}")
        if visualize:
            plt.plot(range(1, epochs + 1), losses, 'b-')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.show()

    def evaluate(self, inputs, targets):

        results = {
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None
        }

        predictions = self.predict(inputs)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        precision, recall, f1_score = F.calculate_classification_metrics(predicted_classes, true_classes, self.num_classes) 

        results["loss"] = self.loss_function(targets, predictions, derivative=False)
        results["accuracy"] = np.mean(predicted_classes == true_classes)
        results["precision"] = precision
        results["recall"] = recall
        results["f1_score"] = f1_score

        return results


       