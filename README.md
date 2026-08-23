# dist-ml-framework

A lightweight, $N$-dimensional tensor library and automatic differentiation engine built from scratch in Python and NumPy.

Inspired by [micrograd](https://github.com/karpathy/micrograd), but scaled up from scalar values to vectorized tensor operations.

## Dependencies

* Python 3.8+
* NumPy
* Matplotlib (for running benchmarks)
* Torchvision (for downloading the MNIST dataset)

```bash
pip install numpy matplotlib torchvision

```

## Quick Example

```python
import numpy as np
from micrograd.multi_dim_engine import Tensor

# Define tensors with gradient tracking
a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = Tensor(np.array([[2.0, 0.0], [1.0, 3.0]]))

# Forward pass
c = a @ b
d = c.relu()
loss = d.sum()

# Backward pass
loss.backward()

print("a grad:\n", a.grad) #prints a grad: [[2. 4.],[2. 4.]]
 
print("b grad:\n", b.grad) #prints b grad: [[4. 4.],[6. 6.]]
 
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

* **The Computational Graph:** As you do operations (`+`, `@`, `relu`), we track them in a directed graph. Calling `.backward()` visits nodes in reverse topological order so every tensor gets its gradients in the exact right sequence.
* **Vectorized Autograd:** Instead of tracking scalar floats one-by-one, we track $N$-dimensional arrays. Matrix multiplications use $A^T$ transpose rules, and batch broadcasting automatically sums gradients back down to match the original tensor shape.
* **Fused Loss Trick:** Softmax and Cross-Entropy are merged into a single clean math step ($\frac{\partial L}{\partial z_i} = p_i - y_i$). We subtract the maximum value before exponentiating so numbers don't explode into `inf` or `NaN`.
