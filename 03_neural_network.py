import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define a simple 2-layer neural network
class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.weights1 = np.random.rand(2, 3)  # 2 input neurons → 3 hidden neurons
        self.weights2 = np.random.rand(3, 1)  # 3 hidden neurons → 1 output neuron
        self.bias1 = np.random.rand(3)
        self.bias2 = np.random.rand(1)

    def forward(self, x):
        hidden_layer = sigmoid(np.dot(x, self.weights1) + self.bias1)
        output_layer = sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2)
        return output_layer

# Example input
network = NeuralNetwork()
X = np.array([1, 0])  # Example input
output = network.forward(X)

print(f"Neural Network Output: {output}")
