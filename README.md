# dist-ml-framework

A minimal, N-dimensional tensor library and automatic differentiation engine built from scratch in Python and NumPy. 

### Overview

`dist-ml-framework` was built from first principles to better understand reverse-mode automatic differentiation, tensor operations, and how computational graphs execute behind the scenes. **Inspired by Andrej Karpathy's scalar-based engine (`micrograd`), this project scales those core autograd principles up to $N$-dimensional tensors with vectorized operations.**

Instead of relying on high-level frameworks like PyTorch or TensorFlow, this library handles graph construction, vectorized backpropagation, and parameter updates directly using custom primitives. 

To verify the mathematical correctness of the engine, it was evaluated on the MNIST digit classification benchmark. Using a custom multi-layer perceptron trained with a fused Softmax Cross-Entropy loss and an Adam optimizer, the framework achieves **>99% test accuracy**, proving the correctness of its gradient propagation and optimization pipeline. 

This repository serves both as an analytical testbed for tensor calculus and as the foundation for an upcoming local RPC-based distributed training module.

---

## 1. Core Architecture & Computational Graph Design

### Graph Construction & Reverse-Mode Autodiff

The framework represents computations as a Directed Acyclic Graph (DAG), where nodes are `Tensor` objects and edges represent differentiable operations. Inspired by the design of Karpathy's `micrograd`, during the forward pass every operation dynamically records its parent tensors and the local backward function required to compute gradients.

To execute backpropagation:

1. The engine constructs a post-order topological sort of the graph starting from the scalar loss node using Depth-First Search (DFS).
2. It processes nodes in reverse topological order, ensuring every parent tensor receives the full sum of its downstream gradients before executing its own backward step:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j \in \text{Children}(i)} \frac{\partial \mathcal{L}}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

Accumulating gradients (`grad += ...`) rather than overwriting them ensures mathematical correctness when a tensor is reused across multiple paths in the computational graph.

## 2. Matrix Calculus & Numerical Stability Protocols

### Vectorized Gradient Derivatives & Broadcasting Rules
Moving from scalar autograd to $N$-dimensional tensors requires tracking shape transformations across operations. For a standard linear layer transformation $Y = XW + B$, where $X \in \mathbb{R}^{B \times D_{in}}$ and $W \in \mathbb{R}^{D_{in} \times D_{out}}$:

- **Weight Gradient:**
  $$\frac{\partial \mathcal{L}}{\partial W} = X^T \cdot \frac{\partial \mathcal{L}}{\partial Y}$$
- **Input Gradient:**
  $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \cdot W^T$$
- **Bias Gradient & Axis Reduction:**
  $$\frac{\partial \mathcal{L}}{\partial B} = \sum_{i=1}^{B} \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, :}$$

Because the bias vector $B \in \mathbb{R}^{1 \times D_{out}}$ is broadcast across the batch dimension $B$, its backward step explicitly sums gradients along the broadcasted axis to preserve shape alignment.

---

### Fused Softmax Cross-Entropy & Overflow Protection
In single-precision floating-point arithmetic (`float32`), computing naive Softmax $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ leads to numerical overflow when logit values exceed $\approx 88.7$.

To guarantee numerical stability, the engine implements a max-shifted log-sum-exp transformation:

$$\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

Furthermore, by fusing the Softmax activation directly with the Categorical Cross-Entropy loss function, the analytical gradient simplifies to:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

where $p_i$ is the predicted probability distribution and $y_i$ is the one-hot target ground truth. Fusing these operations avoids evaluating explicit logarithms of near-zero values ($\log(p) \to -\infty$), completely preventing `NaN` propagation during training.
