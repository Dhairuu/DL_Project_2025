# MNIST Digit Recognizer From Scratch

This project implements a three-layer Neural Network for handwritten digit recognition using the MNIST dataset. The complete neural network—including forward propagation, backpropagation, and gradient descent—is built from scratch using NumPy[file:1].

---

## Features

- **Custom Neural Network:** Feed-forward structure developed with Python and NumPy[file:1].
- **Preprocessing:** Input normalization (`X/255.0`), He Initialization, and ReLU activation for stable training[file:1].
- **Persistent Parameters:** Save/load trained weights and biases via `parameters.json`[file:1].
- **Live Prediction GUI:** Tkinter and Pillow application for real-time digit prediction[file:1].
- **Stable Training:** Step-decay learning rate schedule for optimal convergence[file:1].

---

## Network Architecture

| Layer   | Type   | Size       | Activation | Initialization   |
|---------|--------|------------|------------|------------------|
| Layer 1 | Input  | 784 (28x28)| None       | N/A              |
| Layer 2 | Hidden | 128        | ReLU       | He Initialization|
| Layer 3 | Hidden | 128        | ReLU       | He Initialization|
| Layer 4 | Output | 10         | Softmax    | He Initialization|

---

## Prerequisites

Install required libraries:
