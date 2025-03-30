import numpy as np

# Activation function & derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Mean Squared Error (MSE) loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example loss computation
y_true = np.array([1])  # Expected output
y_pred = np.array([0.5])  # Predicted output from the network

loss = mse_loss(y_true, y_pred)
print(f"Loss: {loss}")

# Example gradient for weight update
gradient = 2 * (y_pred - y_true) * sigmoid_derivative(y_pred)
print(f"Gradient: {gradient}")
