# dist-ml-framework

A minimal, N-dimensional tensor library and automatic differentiation engine built from scratch in Python and NumPy.

### Overview

`dist-ml-framework` was built from first principles to better understand reverse-mode automatic differentiation, tensor operations, and how computational graphs execute behind the scenes. Instead of relying on existing frameworks like PyTorch or TensorFlow, this project handles graph construction, vectorized backpropagation, and parameter updates directly using custom primitives.

To verify the math, the engine was evaluated on the MNIST digit classification benchmark. Using a custom multi-layer perceptron trained with a fused Softmax Cross-Entropy loss and an Adam optimizer, the framework achieves **>99% test accuracy**, proving the correctness of its gradient propagation and optimization pipeline.

This repository serves both as an analytical testbed for tensor calculus and as the foundation for an upcoming local RPC-based distributed training module.



