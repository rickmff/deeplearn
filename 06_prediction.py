import numpy as np

# Sigmoid function
def sigmoid(x): return 1 / (1 + np.exp(-x))

# Trained weights and bias from previous training step
weights = np.array([[4.5], [4.5]])  # Example trained weights
bias = np.array([-6.0])  # Example trained bias

# Test new inputs
X_new = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])

# Make predictions
predictions = sigmoid(np.dot(X_new, weights) + bias)
print(f"Predictions: {predictions}")
