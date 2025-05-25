import numpy as np

# Data Preparation
def one_hot_encoding(column, num_classes, dtype=float):
    if not isinstance(column, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if column.ndim != 1:
        raise ValueError("Input must be a 1 dimensional")

    if column.dtype.kind == "U":
        unique_values = np.unique(column)
        mapped_values = {value: i for i, value in enumerate(unique_values)}
        encoded_values = np.array([mapped_values[value] for value in column])
        column = encoded_values

    return np.eye(num_classes)[column]

# Activation Functions
def sigmoid(x: np.ndarray, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    else:
        return s

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

def softmax(x: np.ndarray, derivative=False):
    if derivative:
        raise NotImplementedError("Softmax derivative is combined with Cross-Entropy loss derivative for now.")
    else:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'softmax': softmax,
    None: None,
}

# Loss Functions
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape of {y_true.shape} and y_pred shape of {y_pred.shape} must match.")
    if derivative:
        return (y_pred - y_true).astype(np.float32)
    else:
        return (np.mean((y_true - y_pred) ** 2)).astype(np.float32)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape of {y_true.shape} and y_pred shape of {y_pred.shape} must match.")
    if derivative:
        return np.sign(y_pred - y_true).astype(np.float32)
    else:
        return np.mean(np.abs(y_true - y_pred)).astype(np.float32)

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, derivative=False):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape of {y_true.shape} and y_pred shape of {y_pred.shape} must match.")
    if derivative: # only combined with SOFTMAX activation function
        return ((y_pred - y_true) / y_true.shape[0]).astype(np.float32)
    else:
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12) # Small epsilon to prevent log(0)
        return (-np.sum(y_true * np.log(y_pred)) / y_true.shape[0]).astype(np.float32)

loss_functions = {
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'cross_entropy': cross_entropy,
    None: None,
}

# Evaluation Helper
def calculate_classification_metrics(predictions, targets, num_classes) :
    current_precisions = []
    current_recalls = []
    current_f1_scores = []

    for c in range(num_classes): # One vs rest if num_classes > 2
        true_positives = np.sum((targets == c) & (predictions == c)) 
        false_positives = np.sum((targets != c) & (predictions == c)) 
        false_negatives = np.sum((targets == c) & (predictions != c)) 

        current_precision = 0
        if (true_positives + false_positives) > 0:
            current_precision = true_positives / (true_positives + false_positives)
        current_precisions.append(current_precision)

        current_recall = 0
        if (true_positives + false_negatives) > 0:
            current_recall = true_positives / (true_positives + false_negatives)
        current_recalls.append(current_recall)

        current_f1 = 0.0
        if (current_precision + current_recall) > 0:
            current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall)
        current_f1_scores.append(current_f1)

    precision = np.mean(current_precisions) if current_precisions else 0
    recall = np.mean(current_recalls) if current_recalls else 0
    f1 = np.mean(current_f1_scores) if current_f1_scores else 0

    return precision, recall, f1

# SKLEARN: Functions to get the number of parameters and RAM usage
import numpy as np

def sklearn_get_num_learnable_params(mlp):
    """Returns the total number of learnable parameters in a scikit-learn MLP."""
    total = 0
    for coef, intercept in zip(mlp.coefs_, mlp.intercepts_):
        total += coef.size + intercept.size
    print(f"Total learnable parameters: {total}")
    return total

def sklearn_get_virtual_ram_usage(mlp, batch_size, dtype=np.float32, training=False):
    """Estimates RAM usage for parameters and activations in a scikit-learn MLP."""
    param_bytes = 0
    for coef, intercept in zip(mlp.coefs_, mlp.intercepts_):
        param_bytes += coef.size * np.dtype(dtype).itemsize
        param_bytes += intercept.size * np.dtype(dtype).itemsize

    layer_sizes = [mlp.coefs_[0].shape[0]] + [coef.shape[1] for coef in mlp.coefs_]
    activation_elements = batch_size * sum(layer_sizes)
    activ_bytes = activation_elements * np.dtype(dtype).itemsize

    if training:
        param_bytes *= 2
        activ_bytes *= 2
    
    total_ram_bytes = param_bytes + activ_bytes

    print(f"Parameter RAM usage: {param_bytes} bytes ({param_bytes / (1024 ** 2):.2f} MB)")
    print(f"Activation RAM usage (batch_size={batch_size}): {activ_bytes} bytes ({activ_bytes / (1024 ** 2):.2f} MB)")
    print(f"Total estimated RAM usage: {total_ram_bytes} bytes ({total_ram_bytes / (1024 ** 2):.2f} MB)")


    return {
        "param_ram_bytes": param_bytes,
        "activ_ram_bytes": activ_bytes,
        "total_ram_bytes": param_bytes + activ_bytes
    }

# PyTorch: Functions to get the number of parameters and RAM usage
import torch

def torch_get_num_learnable_params(model):
    """Returns the total number of learnable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total}")
    return total

def torch_get_virtual_ram_usage(model, batch_size, input_size, dtype=torch.float32, training=False):
    """Estimates RAM usage for parameters and activations in a PyTorch model."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.zeros((batch_size,) + input_size, dtype=dtype)
    activ_elements = dummy_input.numel()
    x = dummy_input
    for layer in model.children():
        try:
            x = layer(x)
            activ_elements += x.numel()
        except Exception:
            continue
    activ_bytes = activ_elements * torch.finfo(dtype).bits // 8

    if training:
        param_bytes *= 2
        activ_bytes *= 2

    total_ram_bytes = param_bytes + activ_bytes

    print(f"Parameter RAM usage: {param_bytes} bytes ({param_bytes / (1024 ** 2):.2f} MB)")
    print(f"Activation RAM usage (batch_size={batch_size}): {activ_bytes} bytes ({activ_bytes / (1024 ** 2):.2f} MB)")
    print(f"Total estimated RAM usage: {total_ram_bytes} bytes ({total_ram_bytes / (1024 ** 2):.2f} MB)")

    return {
        "param_ram_bytes": param_bytes,
        "activ_ram_bytes": activ_bytes,
        "total_ram_bytes": param_bytes + activ_bytes
    }