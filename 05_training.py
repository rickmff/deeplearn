import numpy as np

# Sigmoid function & derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Initialize weights randomly
np.random.seed(42)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
lr = 0.1  # Learning rate

# Training loop
for epoch in range(1000):
    # Forward pass
    hidden = sigmoid(np.dot(X, weights) + bias)

    # Compute loss
    error = y - hidden
    loss = np.mean(error**2)

    # Backpropagation
    adjustments = error * sigmoid_derivative(hidden)
    weights += np.dot(X.T, adjustments) * lr
    bias += np.sum(adjustments) * lr

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")
