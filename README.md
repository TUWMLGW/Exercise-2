# Exercise 2: Neural Network Implementation & Comparative Analysis

This project explores and compares three approaches to building neural networks:

1. **From Scratch:**  
   Implementing a neural network using only Python and NumPy, with all forward and backward propagation coded manually.

2. **PyTorch:**  
   Building and training neural networks using the PyTorch deep learning framework.

3. **LLM-Generated:**  
   Using a Large Language Model (LLM) to generate neural network architectures, which are then implemented and trained.

Each approach is applied to two distinct datasets, experimenting with different hyperparameters and network depths to analyze performance and flexibility.

---

## 📝 Project Goals

- **Understand** the inner workings of neural networks by building one from scratch.
- **Leverage** PyTorch for efficient model development and training.
- **Explore** the capabilities of LLMs in generating viable neural network architectures.
- **Compare** the performance and flexibility of each approach on different datasets and with various hyperparameters.

---

## 🤯 How to Use the Custom NN from Scratch

1. **Define Layers**

```python
from nn_from_scratch.layer import Layer

layers = [
    Layer(input_size=4, output_size=8, activation_function='relu'),
    Layer(input_size=8, output_size=3, activation_function='softmax')
]
```
2. **Initialize the Neural Network**
```python
from nn_from_scratch.nn import NN

model = NN(
    layers=layers,
    num_classes=3,
    activation_function='relu',      # Default for layers without explicit activation
    loss_function='cross_entropy'    # Must be defined in functions.py
)
```
4. **Train the Model**
```python
model.train(
    inputs=X_train,
    targets=y_train,
    epochs=100,
    learning_rate=0.01,
    batch_size=32,
    verbose=True,
    visualize=True
)
```
6. **Make Predictions**
```python
predictions = model.predict(X_test)
```
8. **Evaluate**
```python
model.evaluate(X_test, y_test)
```
10. **Utilities**
```python
model.get_num_learnable_params()
model.get_virtual_ram_usage(batch_size=32, training=True)
```

*Made for the Machine Learning course, TU, Summer Semester 2025.*
