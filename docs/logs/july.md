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

### 📌 July 13 & 14, 2026 — Multiple Variables, Vectorization, and Feature Scaling
* **What I did:** We learned how to expand our linear regression model to handle multiple variables (features) instead of just one. Now, every single input feature has its own weight. I also studied "feature scaling" because when features have totally different ranges, gradient descent becomes incredibly slow. I learned three ways to fix this and keep values roughly between -1 and 1: Min-Max Scaling, Mean Normalization, and Z-score Normalization.
* **The Breakthrough:** I implemented both a loop-based calculation and a vectorized calculation using a dot product. Comparing them side-by-side was eye-opening. The vectorized code is not only much shorter and cleaner to read, but it runs insanely faster because it processes the data in parallel instead of waiting for loops to finish.
* **Next Step:** Apply these scaling techniques to a real dataset and practice writing clean, vectorized code using NumPy.

## Practical Project: Sales Prediction
To apply these concepts, I built a model in the `/playground` folder, `ml_adver_pred.py` to predict product units sold based on advertising budgets for TV, Radio, and Newspaper.

### Problems & Solutions:
*   **The Issue:** On my first attempt, the gradient descent failed. The cost function diverged and went to infinity because the TV and Newspaper budget numbers were too large for my learning rate.
*   **The Fix:** I applied **Z-score normalization** to the dataset. Once the features were scaled, the gradient descent stabilized and converged smoothly.
*   **The Result:** The model successfully trained. The final weights showed that TV advertising had the highest weight, meaning it has the strongest impact on sales.

### 📌 July 15, 2026 — Course Completion & Connecting the Dots
* **What I did:** I officially finished the first course on Regression and Classification by Andrew Ng. In the final modules, I learned about logistic regression, the sigmoid function, and how to look at classification outputs. I also studied decision boundaries, how feature engineering helps the model fit data better, and how to deal with underfitting and overfitting using the cost function and gradient descent.
* **The Breakthrough:** Finishing this course made a huge difference. I tried watching Andrej Karpathy's video (*The Spelled-out Intro to Neural Networks and Backpropagation*) before, but parts of it were confusing. Today, the first few minutes finally clicked in my mind because I actually understood the underlying math. 
* **Next Step:** I am going to watch as much of Karpathy's video as I can today to see how this works in code. After that, I will finish the next two courses in Andrew Ng's specialization, and then come back to Karpathy's video to rewatch it until I completely master backpropagation.

### 📌 July 16, 2026 — Diving into Neural Networks & Layer Visualization
* **What I did:** I started the second course in the specialization, *Advanced Learning Algorithms*. I got introduced to the history of neural networks and how they were originally inspired by the neurons in the human brain. I learned how data travels through layers with multiple nodes, where every single node has its own weights and bias. 
* **The Breakthrough:** Understanding how layers build on top of each other was really cool. Using the example of an image, I saw how the first layer only detects tiny lines and edges. The next layer takes those lines and detects small shapes, the following layer groups those shapes together, and the final layer uses all of that to give the final output. 
* **Next Step:** I want to download the pre-trained weights and biases of an existing neural network from the internet, write a script to feed it new data, and see how the network processes it.

### 📌 July 17, 2026 — Pivoting from TensorFlow to First-Principles from Scratch
* **What I did:** I was continuing Week 1 of the *Advanced Learning Algorithms* course by Andrew Ng, but I noticed he started using TensorFlow and showing PyTorch as the alternative. Since high-level frameworks like TensorFlow and PyTorch are completely banned in my project (because my goal is to build a mini-PyTorch entirely from scratch using only pure Python and NumPy), I decided to stop this course. 
* **The New Strategy:** I shifted my strategy and moved over to Andrej Karpathy's *Zero to Hero* playlist on YouTube. His course is exactly what I need because it focuses on building everything from absolute first principles.
* **Next Step:** Dive deep into Karpathy's first video to study how backpropagation works at a scalar level and start mapping out my own custom autograd engine.

### 📌 July 18, 2026 — Coding Autograd from Scratch & Deep Dives
* **What I did:** I restarted Andrej Karpathy's first video and began coding the autograd engine alongside him. I made a rule for myself not to type a single line of code until I understood it 100%. To truly visualize how the mathematical chains connect, I integrated the graph visualization code instead of blindly typing out the logic.
* **Problems & Solutions:**
  * **Problem 1:** I was confused why he used `_children` and `_op` with an underscore inside the `__init__` arguments.
    * **Solution:** I researched it and found it's a standard Python developer convention. It tells other programmers that these variables are internal and meant for the backend, not for the end-user to interact with directly.
  * **Problem 2:** I noticed he initialized the children argument as an empty tuple `()` and then converted it to a set, rather than just using `{}` directly.
    * **Solution:** I realized that in Python, using `{}` defaults to creating an empty dictionary, not a set. Passing a tuple and casting it to a set is the correct way to handle it.
* **Next Step:** Continue building out the forward pass of the expression graph and start preparing for the actual backpropagation implementation.

### 📌 July 19 & 20, 2026 — Deep Dive into Closures, Scope, and Topological Sorting
* **July 19:** Off day.
* **July 20:** Continued Andrej Karpathy's first video on autograd. Sticking strictly to my rule of not writing a single line of code until I understand it 100%. This takes a lot of extra time because a 1-minute clip often leads to an hour of researching Python prerequisites, but it ensures I truly master the mechanics.
* **Problems & Solutions:**
  * **Problem 1:** I didn't understand why `lambda` was being used in some places.
    * **Solution:** I researched Python anonymous functions and learned that `lambda` allows you to create quick, inline functions (or no-op placeholder functions) without a full `def` block.
  * **Problem 2:** The assignment `out._backward = _backward` confused me. Why were there no parentheses, and why didn't the inner `_backward()` function take `self` as an argument when defined inside `__add__` or `__mul__`, yet called `out._backward()` with parentheses later?
    * **Solution:** I did a deep dive into Python closures and scope. In Python, functions are first-class objects—assigning `out._backward = _backward` without parentheses stores the function reference itself to be executed later. Additionally, an inner function defined inside a method automatically inherits access to all variables in that method's scope (a closure), so it doesn't need `self` in its definition. However, when defining `backward(self)` at the class level outside those methods, `self` is explicitly required to access instance attributes.
  * **Problem 3:** The recursive topological sort algorithm for ordering the execution graph was really mind-bending.
    * **Solution:** I spent time tracing the recursion step-by-step. I realized recursion works like opening a window inside a window. You keep opening deeper windows until you can't go any further (the leaf nodes), complete the work in the deepest window, close it, step back to the previous window, and repeat that process until you return to the root.
* **Next Step:** Finish building the automated backward pass for the complete expression graph and test it against a simple neural network layer.