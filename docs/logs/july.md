### 📝 Dev Log: July 8, 2026 – Project Inception & Math Core

#### **Overview**
Today marks the official start of building my custom distributed machine learning micro-framework from scratch. The goal of this session was to establish a solid object-oriented foundation for handling mathematical tensors.

#### **What I Accomplished**
1. **Repository Setup:** Initialized the project repository and structured the initial workspace directory.
2. **Created `Tensor.py`:** Created the core script that will act as the baseline execution engine for all neural network operations.
3. **Core Mathematical Operators:** Successfully implemented key Python dunder methods (`__add__`, `__sub__`, `__mul__`, `__truediv__`, and `__pow__`) to support foundational matrix calculations.

---

#### **Challenges & Solutions Encountered**

*   **Problem 1: Dependency Management & Global Environment Conflicts**
    *   *Issue:* Ran into package isolation conflicts while attempting to install and configure NumPy globally on my system.
    *   *Solution:* Implemented local environment isolation by setting up and activating a Python virtual environment (`.venv`). This allowed for a clean, secure dependency installation dedicated solely to this project.
*   **Problem 2: Operation Chaining & Variable Type Mismatches**
    *   *Issue:* While attempting to chain operations (e.g., adding multiple arrays sequentially), the program crashed. The first operation was returning a raw NumPy array instead of a custom class object, making subsequent operations impossible.
    *   *Solution:* Refactored the core architecture of all magic methods. Instead of returning the raw array data directly, every operation now dynamically bundles the mathematical output into a brand new `Tensor` object. This successfully enabled clean, infinite method chaining.
*   **Problem 3: Numerical Instability (Division by Zero)**
    *   *Issue:* Standard vector division exposed the framework to potential runtime crashes or infinite math loops (`inf` / `nan`) if a divisor array contained a zero.
    *   *Solution:* Introduced an out-of-place numerical stabilizer using epsilon ($\epsilon = 10^{-8}$). By adding this microscopic scalar to the denominator during the division step, the engine remains mathematically stable without mutating the original matrix values.

### 📌 July 11, 2026 — Foundational Machine Learning & Linear Regression Math
* **Milestone:** Began the *Supervised Machine Learning: Regression and Classification* course by Andrew Ng to build the theoretical foundation for the micro-framework.
* **Core Concepts Covered:**
  * **Supervised vs. Unsupervised Learning:** Differentiated between training models with explicit target outputs versus finding underlying patterns and relations in unlabeled data.
  * **Regression vs. Classification:** Mapped the difference between continuous infinite outputs (regression) and discrete, specific categories (classification).
  * **Linear Regression Mechanics:** Studied the single-variable linear model $f(x) = wx + b$.
  * **Cost Function & Optimization:** Analyzed the Mean Squared Error cost function to calculate total error between predictions and actual targets.
  * **Gradient Descent Algorithm:** Explored the mathematical optimization process used to update the parameters $w$ (weight) and $b$ (bias) iteratively using a learning rate ($\alpha$) until they converge at a local minimum of the cost function.
* **Next Step:** Complete the remaining modules on gradient descent implementation and begin translating these mathematical formulas into raw NumPy code.