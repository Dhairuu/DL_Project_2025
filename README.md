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
```
pip install pandas numpy Pillow matplotlib
```


You'll need the `train.csv` and `test.csv` files from the official Kaggle MNIST competition for training and validation[file:1].

---

## Project Files

| File             | Description                                                                                     |
|------------------|------------------------------------------------------------------------------------------------|
| Training.py      | Script to train the network (data loading, network, training loop, save parameters)             |
| Testing.py       | Script to load trained parameters and test model accuracy on `test.csv`                         |
| Utils.py         | Core logic (layer classes, activation functions, one-hot encoding, forward/backward propagation)|
| LivePredict.py   | Tkinter/PIL GUI for drawing and live prediction                                                 |
| parameters.json  | Stores trained weights (W) and biases (B) for layers 2, 3, and 4                               |

---

## Usage

### Step 1: Train the Model

Before testing or using live prediction, train the model and generate parameters:

```
python Testing.py
```


### Step 3: Live Digit Recognition

Start the interactive drawing application:

```
python LivePredict.py
```

- A `1000x1000` canvas window will open.
- Draw any digit (0–9), then click **Predict**[file:1].
- Application downscales input to `28x28`, runs the model, and predicts the digit[file:1].

---

## License

[Add your license information here]

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss major changes[file:1].
