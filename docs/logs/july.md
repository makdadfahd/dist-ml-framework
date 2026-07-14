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

### July 13 / 14, 2026 — 2 days Summary & Project Implementation

## 1. Concepts Learned

### Multiple Linear Regression & Vectorization
I moved from using one variable to multiple variables. In this setup, every input feature has its own weight. 

Instead of using loops to calculate predictions:

$$f_{\vec{w},b}(\vec{x}) = \sum_{j=1}^{n} w_j x_j + b$$

I implemented **vectorization**. This reduces the calculation to a simple dot product:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

Vectorization makes the code much shorter, cleaner, and faster. I compared a loop-based implementation with the vectorized version, and the vectorized code runs significantly faster because it processes data in parallel.

### Feature Scaling
When features have wildly different ranges (some too large, some too small), gradient descent becomes highly inefficient. To fix this, we scale the features so their values lie roughly between -1 and 1. I studied three methods:

1.  **Min-Max Scaling (Feature Scaling):** Rescaling data to fit between 0 and 1.
2.  **Mean Normalization:** Centering the data around zero: $x_i \leftarrow \frac{x_i - \mu_i}{\max - \min}$.
3.  **Z-score Normalization:** Scaling using the mean and standard deviation: $x_i \leftarrow \frac{x_i - \mu_i}{\sigma_i}$.

### Feature Engineering
I learned how to create new features by transforming or combining the original ones. This helps the model learn complex patterns without changing the linear algorithm itself.

---

## 2. Practical Project: Sales Prediction
To apply these concepts, I built a model in the `/playground` folder to predict product units sold based on advertising budgets for TV, Radio, and Newspaper.

### Problems & Solutions:
*   **The Issue:** On my first attempt, the gradient descent failed. The cost function diverged and went to infinity because the TV and Newspaper budget numbers were too large for my learning rate.
*   **The Fix:** I applied **Z-score normalization** to the dataset. Once the features were scaled, the gradient descent stabilized and converged smoothly.
*   **The Result:** The model successfully trained. The final weights showed that TV advertising had the highest weight, meaning it has the strongest impact on sales.

