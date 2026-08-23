# dist-ml-framework

A lightweight, $N$-dimensional tensor library and automatic differentiation engine built from scratch in Python and NumPy.

Inspired by [micrograd](https://github.com/karpathy/micrograd), but scaled up from scalar values to vectorized tensor operations.

## Dependencies

* Python 3.8+
* NumPy
* Matplotlib (for running benchmarks)

```bash
pip install numpy matplotlib

```

## Quick Example

```python
import numpy as np
from micrograd.multi_dim_engine import Tensor

# Define tensors with gradient tracking
a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
b = Tensor(np.array([[2.0, 0.0], [1.0, 3.0]]), requires_grad=True)

# Forward pass
c = a @ b
d = c.relu()
loss = d.sum()

# Backward pass
loss.backward()

print("a grad:\n", a.grad)
print("b grad:\n", b.grad)

```

## Running MNIST

We evaluated a 3-layer MLP ($784 \to 128 \to 64 \to 10$) trained with custom Adam optimizer and fused Softmax Cross-Entropy loss across 10 epochs.

```bash
python micrograd/experiments/compare_activations.py

```

### Activation Benchmarks
![Activation Benchmarks](activation_benchmark.png)

* **ReLU:** **97.84%** test accuracy (Final Loss: `0.0193`)
* **Tanh:** **97.72%** test accuracy (Final Loss: `0.0175`)
* **Sigmoid:** **97.57%** test accuracy (Final Loss: `0.0494`)

## Under the Hood

* **Autograd DAG:** Constructs computational graph dynamically and performs backpropagation using post-order topological sort.
* **Vectorized Operations:** Supports matrix calculus derivatives and automatic bias reduction across broadcasted batch dimensions.
* **Fused Loss:** Combines Softmax and Cross-Entropy into a stable analytical gradient ($\frac{\partial L}{\partial z_i} = p_i - y_i$) with max-shifting to prevent overflow.
