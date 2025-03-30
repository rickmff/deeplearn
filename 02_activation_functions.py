import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)

# Generate inputs
x = np.linspace(-5, 5, 100)

# Plot activations
plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid(x), label="Sigmoid", color="blue")
plt.plot(x, relu(x), label="ReLU", color="red")
plt.plot(x, tanh(x), label="Tanh", color="green")

plt.legend()
plt.title("Activation Functions")
plt.show()
