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

### 📌 August 7, 2026 — Matrix Multiplication (`__matmul__`, `__rmatmul__`) & Dimension Handling
* **What I did:** Added matrix multiplication (`__matmul__` and `__rmatmul__`) to the tensor engine. 
* **Problems & Solutions:**
  * **1D Array Reshaping in `__matmul__`:** 
    * **Problem:** Multiplying 1D NumPy vectors caused shape mismatch errors during matrix operations.
    * **Solution:** Ensured 1D arrays are reshaped into 2D matrices so matrix multiplication dimensions stay clean and predictable.
  * **Handling Non-Commutativity in `__rmatmul__`:**
    * **Problem:** When trying to evaluate `array @ tensor`, I initially returned `self @ other` inside `__rmatmul__` and got shape errors.
    * **Solution:** Matrix multiplication is non-commutative A @ B != B @ A. Because `array @ tensor` triggers `tensor.__rmatmul__(array)`, the correct operand order is `other @ self`.
  * **Implicit Type Coercion:**
    * **Problem:** I wondered why `self = Tensor(self)` wasn't working inside `__matmul__` when `self` wasn't a `Tensor`.
    * **Solution:** I realized that if `self` isn't a `Tensor`, execution never hits `__matmul__` in the first place—Python routes straight to `tensor.__rmatmul__(array)`. Wrapping `other` via `other = other if isinstance(other, Tensor) else Tensor(other)` and returning `other @ self` solved it cleanly.
* **Takeaway on Backprop:** Stanford CS231n Lecture 4 stayed high-level and didn't detail multi-dimensional gradient mechanics, so I am leaving `_backward()` methods empty for now until I fully map out the matrix calculus rules.
* **Next Step:** Study matrix calculus shape rules (transposes, shape matching, and sum reductions across batch/broadcast axes) to implement `_backward()` for `__matmul__`, `__add__`, and `__mul__`.

### 📌 August 8, 2026 — Matrix Calculus Derivation & 1D Vector Dimension Handling
* **What I did:** Found a detailed video tutorial deriving the matrix calculus rules for matrix multiplication backpropagation, enabling me to begin writing `_backward()` for the `__matmul__` method.
* **Problems & Solutions:**
  * **1D Vector Reshaping during Forward Pass:**
    * **Problem:** In mathematical notation, intuition naturally treats 1D vectors as row or column matrices during multiplication, but NumPy throws strict rank/dimension mismatch errors.
    * **Solution:** Added explicit reshaping based on operand positioning before executing the matrix multiplication:
      * Left operand (`self`): Reshaped to 2D row vector `(1, -1)`.
      * Right operand (`other`): Reshaped to 2D column vector `(-1, 1)`.
  * **Output Dimension Alignment & Rank Squeezing:**
    * **Problem:** Applying rigid 2D matrix multiplication rules consistently forces the output into a 2D matrix, even when multiplying 1D vectors.
    * **Solution:** Implemented dynamic dimension squeezing on the resulting matrix shape:
      * **Both operands are 1D:** Squeeze all dimensions (`squeeze()`) to return a scalar.
      * **Left operand (`self`) is 1D:** Squeeze the first dimension (`squeeze(0)`).
      * **Right operand (`other`) is 1D:** Squeeze the last dimension (`squeeze(-1)`).
* **Next Step:** Apply these derived matrix rules to complete the backward pass logic ($\frac{\partial L}{\partial A} = \text{grad} \cdot B^T$ and $\frac{\partial L}{\partial B} = A^T \cdot \text{grad}$) while maintaining shape alignment across transposes.

### 📌 August 9–11, 2026 — Matrix Autograd `_backward()` & Gradient Dimension Alignments
* **August 9–10:** Off days.
* **August 11 (Today):** Implemented the `_backward()` pass for matrix multiplication (`__matmul__`) and added the initial backward logic for addition (`__add__`).
* **Problems & Solutions:**
  * **Raw Gradient Operations in `_backward()`:**
    * **Problem:** I initially wrote `self.grad = out.grad @ w.data.T`, but because `out.grad` and `w.data.T` are raw internal data structures rather than custom `Tensor` objects, the 1D/2D dimension rules I wrote inside `__matmul__` didn't execute, causing shape mismatches.
    * **Solution:** Re-implemented matrix dimension handling directly inside the `_backward()` function. I wrapped all participating gradient arrays into explicit 2D matrices for the transpose matrix multiplication, calculated the resulting gradients, and then reshaped/squeezed the resulting arrays back so that `self.grad` and `other.grad` strictly match the original shapes of `self.data` and `other.data`.
* **Next Step:** Fine-tune `__add__` backward to handle NumPy broadcasting reduction (summing gradients along broadcast axes) and test the full backward pass end-to-end.

### 📌 August 12, 2026 — The N-Dimensional Tensor Engine is Finally Complete!

Yesterday and today were definitely tough. Figuring out the backward pass for `__add__` and `__mul__` ended up being way harder than `__matmul__`. Matrix multiplication was mostly straightforward matrix calculus, but addition and multiplication—which were almost trivial in the scalar engine—turned out to be a whole different monster in multi-dimensional space because of NumPy broadcasting.

* **The Broadcasting Backprop Wall:**
  * **The Issue:** When you add a tensor of shape `(3,)` to a tensor of shape `(100, 3)`, NumPy automatically broadcasts the smaller one to match `(100, 3)`. The output gradient comes back as `(100, 3)`. Trying to pass that gradient straight back to the `(3,)` leaf node immediately crashes with a shape mismatch.
  * **The Breakthrough:** To undo broadcasting during backprop, you have to sum up the gradients along every axis that got expanded during the forward pass. Saying it in plain English is easy, but turning that logic into bug-free Python code took some serious head-scratching.

* **How I Solved It (`unbroadcast` helper):**
  * Built a dedicated `unbroadcast` function that takes `out.grad` and the targeted target shape (`self.data.shape` or `other.data.shape`).
  * Calculated the rank difference and prepended `1`s to the left of the target shape until both shapes had the same number of dimensions.
  * Loop through paired axes `(m, n)` using `enumerate` and `zip`. Whenever `m > 1` and `n == 1`, it means that axis was stretched in the forward pass. I summed across `axis=i` keeping `keepdims=True` so we don't drop rank prematurely.
  * Finally, reshaped the array back to the original input shape. Applied this exact same reduction logic to `__mul__`.

* **Final Engine Tweaks & Benchmark:**
  * **Topological Seed:** Changed the root gradient initialization from Karpathy’s scalar `self.grad = 1` to `self.grad = np.ones(self.data.shape)`.
  * **Power Rule:** Implemented `_backward()` for `__pow__` using the standard calculus power rule applied across array elements.
  * **Sanity Check:** Built a test script with multi-dimensional array inputs, ran a heavy chain of operations, and hit `.backward()`. Every single gradient from the root down to the leaf nodes calculated cleanly on the first shot!
