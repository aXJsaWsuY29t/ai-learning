# Project 02 — Neural Network & Backpropagation

## Goal

Implement a multi-layer perceptron (MLP) completely from scratch — including the forward pass, loss function, and manual backpropagation. No autograd, no frameworks.

This is the hardest project in the series. Take your time here. If you understand backprop, everything else becomes mechanical.

---

## Why This Matters

Every neural network — including transformers — learns by computing gradients and adjusting weights. If you don't understand how gradients flow backward through a network, the training loop of any model will feel like magic.

---

## Concepts

- **Forward pass** — computing predictions layer by layer
- **Activation functions** — ReLU, sigmoid, tanh and their derivatives
- **Loss function** — measuring how wrong the model is (MSE, cross-entropy)
- **Backpropagation** — chain rule applied layer by layer to compute gradients
- **Gradient descent** — updating weights in the direction that reduces loss
- **Vanishing/exploding gradients** — why they happen and how to spot them

---

## Tasks

### 1. Single Neuron
- Implement a single neuron: `output = activation(dot(weights, input) + bias)`
- Implement sigmoid and its derivative
- Manually compute the gradient of loss w.r.t. weight for one example

### 2. Two-Layer MLP
- Implement `forward(X)` for a 2-layer network (input → hidden → output)
- Use ReLU for hidden layer, sigmoid for output
- Implement MSE loss

### 3. Manual Backpropagation
- Derive and implement gradients for each layer by hand
- No autograd — compute `dL/dW2`, `dL/db2`, `dL/dW1`, `dL/db1` explicitly
- Add gradient checking: compare your gradients to numerical gradients

### 4. Training Loop
- Implement gradient descent update: `W = W - lr * dL/dW`
- Train on XOR problem (4 examples, binary classification)
- Plot loss over epochs — it should decrease

### 5. Deeper Network (optional but recommended)
- Extend to 3 layers
- Observe vanishing gradients with sigmoid — switch to ReLU and compare
- Train on a slightly larger dataset (e.g., 2D spiral classification)

---

## Correctness Goals

- Gradient check passes: manual gradients match numerical gradients within 1e-5
- XOR problem converges to loss < 0.01 within 10,000 epochs
- Loss curve is monotonically decreasing (or nearly so) with a good learning rate
- You can explain what `dL/dW` means in plain English

---

## Key Insight to Internalize

Backprop is just the chain rule applied repeatedly:

```
dL/dW1 = dL/dout * dout/dhidden * dhidden/dW1
```

Each layer passes its gradient to the layer before it. That's it.

---

## Format

Plain `.py` files. The math is the focus — keep code minimal and readable.

```
02-neural-network-backprop/
├── README.md
├── neuron.py          ← single neuron + sigmoid
├── mlp.py             ← 2-layer MLP, forward + backprop
├── train_xor.py       ← training loop on XOR
└── gradient_check.py  ← numerical gradient verification
```

---

## Next Step

Once your XOR network trains and your gradient check passes, move to **Project 03**.
