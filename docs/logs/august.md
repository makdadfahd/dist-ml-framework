### 📌 August 1, 2026 — Scalar Autograd Engine Completed & Pushed
* **What I did:** Spent the entire morning rewatching Andrej Karpathy's first *Zero to Hero* video. Even though I have watched it several times before, every single rewatch reveals new details. After finishing the engine section of the video, I built out all the necessary remaining functions and officially pushed the completed scalar version of `engine.py` to GitHub!
* **Deep Dives & Insights:**
  * **Python `__rmul__` Mechanics:** I finally understood how reverse operations work. If you try to evaluate `2 * a`, Python first attempts `(2).__mul__(a)`. Since an integer doesn't know how to multiply a custom `Value` object, it fails and falls back to `a.__rmul__(2)`, which evaluates `a * 2` instead. I also saw how this applies to true division (`__truediv__`).
  * **Using `assert` for Guards:** I realized why `assert` was used in the `__pow__` method instead of manual `if`/`else` blocks. `assert` acts as a quick guard clause—it guarantees that power operations are restricted to integer/float exponents right at the start, throwing an error immediately if the condition isn't met.
* **Next Step:** Start studying Stanford CS231n (Lectures 3 & 4) to understand how gradients flow through multi-dimensional arrays, then prepare to upgrade the engine to handle NumPy arrays.

### 📌 August 2, 2026 — First MLP Neural Network Built & Import Resolving
* **What I did:** Built and pushed the first completed version of a simple Multi-Layer Perceptron (`MLP(3, [2, 3, 1])`) using my scalar autograd engine. The network takes 3 inputs, feeds them through two hidden layers (2 nodes and 3 nodes), and outputs a single value. It still needs some fine-tuning, but the forward pass and structure are working!
* **Problems & Solutions:**
  * **Problem:** I kept running into module errors when trying to call methods on my custom class (like `engine.Value.tanh()` or trying to access internal data like `engine.Value.loss.data`).
  * **Solution:** I realized the issue was how I imported the module. I had written `from engine import Value`, which brought `Value` directly into the file's namespace, making the prefix `engine.` invalid. Changing the import approach or accessing attributes directly on instances of `Value` fixed the references.
* **Next Step:** Implement a simple loss function and training loop to optimize the MLP's weights and biases via gradient descent.

### 📌 August 3, 2026 — Mini Neural Network Optimization & Regularization Deep Dive
* **What I did:** Built and ran a complete training loop for a mini neural network using my custom scalar autograd engine. I am also watching Lecture 3 of Stanford's CS231n to study loss functions and optimization.
* **Achievements & Breakthroughs:**
  * **Successful Training Loop:** My scalar engine successfully propagated gradients backward and updated the weights, dropping the total loss from **4.5 down to 0.00063**. Seeing the loss converge proved that my forward and backward autograd logic is working correctly.
  * **Understanding Regularization:** I finally had a complete "aha!" moment regarding regularization (L1/L2 penalties). I had first encountered it two weeks ago in Andrew Ng's course, but watching CS231n made it click: regularization acts as a barrier or penalty against overly large weights, forcing the network to distribute its attention across features rather than memorizing noise (overfitting).
* **Next Step:** Complete watching CS231n Lecture 3 and then move on to  Lecture 4 to study matrix calculus and backpropagation through vector/tensor operations, preparing to extend my engine to NumPy arrays.

### 📌 August 4–6, 2026 — $N$-Dimensional Tensor Upgrade & ReLU Migration
* **August 4–5:** Off days.
* **August 6 (Today):** Started upgrading my scalar engine to handle multi-dimensional tensors/NumPy arrays. I began with fundamental operations: `__repr__`, `__add__`, and `__mul__`.
* **Deep Dives & Performance Optimization:**
  * **Replacing `tanh` with `ReLU`:** I removed the `tanh` activation function and replaced it with `ReLU`. Computing $e^x$ in `tanh` requires expensive floating-point operations on the CPU/GPU, whereas `ReLU` is just a simple thresholding operation ($\max(0, x)$). Additionally, research (e.g., AlexNet) shows `ReLU` allows neural networks to converge up to 6 times faster than `tanh` while avoiding vanishing gradients.
  * **Tensor Broadcast Verification:** Successfully tested tensor-with-tensor and tensor-with-scalar additions and multiplications to ensure basic NumPy broadcasting rules behave as expected.
* **Next Step:** Implement shape tracking, matrix multiplication (`__matmul__`), and axis reduction during the backward pass to handle matrix gradient propagation cleanly.