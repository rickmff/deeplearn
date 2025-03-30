import numpy as np

# Activation function: Step function (perceptron)
def step_function(x):
    return 1 if x >= 0 else 0

# Inputs and weights
inputs = np.array([1, 0])  # Example input
weights = np.array([0.5, -0.6])  # Example weights
bias = 0.1  # Example bias

# Neuron calculation
output = step_function(np.dot(weights, inputs) + bias)

print(f"Neuron Output: {output}")
