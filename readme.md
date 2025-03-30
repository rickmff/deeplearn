# Neural Network Simulation with Python

## Overview
This project simulates how a **neuron** works in deep learning using Python. It provides a step-by-step implementation of fundamental neural network concepts, from basic mathematical operations to a simple **PyTorch-based Large Language Model (LLM)**.

| Step | Concept | File Name |
|------|---------|-----------|
| 1️⃣  | Neuron (Perceptron) Calculation | 01_neuron.py |
| 2️⃣  | Activation Functions (Sigmoid, ReLU, Tanh) | 02_activation_functions.py |
| 3️⃣  | Building a Simple Neural Network (Forward Pass) | 03_neural_network.py |
| 4️⃣  | Loss Function & Backpropagation | 04_backpropagation.py |
| 5️⃣  | Training the Neural Network (Gradient Descent) | 05_training.py |
| 6️⃣  | Making Predictions on New Data | 06_prediction.py |

## Requirements
Ensure you have the required Python libraries installed:
```sh
pip install numpy matplotlib torch torchvision torchaudio transformers datasets fastapi uvicorn
```

## Files & Descriptions

### 1️⃣ `01_neuron_activation.py` – Simulating a Neuron
**Concept:** Implements a basic artificial neuron with an activation function.

**What it does:**
- Defines a simple **neuron equation**: `output = activation_function(weight * input + bias)`.
- Uses **Sigmoid** and **ReLU** as activation functions.
- Plots the **activation function curve** using `matplotlib`.

**Usage:**
```sh
python 01_neuron_activation.py
```

---

### 2️⃣ `02_perceptron.py` – Building a Simple Perceptron
**Concept:** Implements a **single-layer perceptron**, the fundamental unit of neural networks.

**What it does:**
- Uses **NumPy** to perform matrix operations.
- Implements a simple perceptron that learns to classify **AND / OR / XOR** logic gates.
- Trains the perceptron using **gradient descent**.

**Usage:**
```sh
python 02_perceptron.py
```

---

### 3️⃣ `03_multilayer_perceptron.py` – Implementing an MLP
**Concept:** Expands on the perceptron to create a **multi-layer perceptron (MLP)**.

**What it does:**
- Defines a **feedforward neural network** with multiple layers.
- Uses **backpropagation** to update weights.
- Simulates training on a **simple dataset**.

**Usage:**
```sh
python 03_multilayer_perceptron.py
```

---

### 4️⃣ `04_neural_network_training.py` – Training a Neural Network
**Concept:** Trains an MLP model on real data.

**What it does:**
- Uses **PyTorch** to create a small neural network.
- Loads a **sample dataset**.
- Implements a **training loop** to optimize model performance.
- Saves the trained model.

**Usage:**
```sh
python 04_neural_network_training.py
```

---

### 5️⃣ `05_visualize_training.py` – Visualizing Training Progress
**Concept:** Helps users understand how training progresses over time.

**What it does:**
- Plots the **loss function** over multiple epochs.
- Visualizes **how weights change** during training.
- Uses `matplotlib` for real-time visualization.

**Usage:**
```sh
python 05_visualize_training.py
```

---

### 6️⃣ `06_neural_network_inference.py` – Using a Trained Model
**Concept:** Demonstrates how to use a trained neural network.

**What it does:**
- Loads a **pre-trained neural network**.
- Uses the model to make **predictions** on new input data.
- Prints and visualizes the output.

**Usage:**
```sh
python 06_neural_network_inference.py
```

---

## Next Steps (wip...)
Once you understand these fundamental concepts, you can move to:

[ ]- `07_simple_llm.py` – Creating a **transformer-based LLM**.
[ ]- `08_train_llm.py` – Training an LLM with real text data.
[ ]- `09_use_llm.py` – Using the LLM for text generation.
[ ]- `10_deploy_llm.py` – Deploying the model as a **FastAPI API**.

## Conclusion
This project provides a structured **step-by-step guide** to understanding **neural networks** and **deep learning models**. By running these files sequentially, you'll gain hands-on experience with fundamental **machine learning concepts**.

---
**Author:** Henrique Faria
**License:** MIT