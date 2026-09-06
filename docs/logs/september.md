### 📌 September 1, 2026 — Network Fundamentals, Socket Programming, & Building a Simple Chatroom

Yesterday I spent the whole day learning about the `socket` library. At first, I started watching tutorial videos on setting up server-client connections, but I stopped because I was just memorizing syntax without actually understanding what the code was doing. 

I switched to a 2-hour video on networking fundamentals that explained everything from scratch, and it made the code click completely.

* **Understanding Sockets & Protocols (The Mental Model):**
  * **The Analogy:** A socket connection is like two tin cans connected by a string. The server holds one can, the client holds the other, and the string is the network connection.
  * **Creating the Socket:** We set up a server socket using `socket.socket(socket.AF_INET, socket.SOCK_STREAM)`. `AF_INET` means using IPv4 addresses, and `SOCK_STREAM` specifies the TCP protocol.
  * **TCP vs. UDP:**
    * **TCP (Phone Call):** Requires a connection before sending data. The server says "hello, do you hear me?", the client responds "yes", data is transferred reliably, and then they hang up. If a packet is lost, it gets resent.
    * **UDP (Throwing a Rock with a Note):** Doesn't wait for a connection or guarantee delivery. You throw the rock at a window and hope your friend gets it.
    * **Why TCP for my project:** For my distributed engine, I **must** use TCP. Gradients and weights cannot afford missing data—if a single gradient tensor gets dropped, the entire backpropagation math breaks.

* **How the Server-Client Lifecycle Works:**
  * **Bind & Listen:** `bind()` assigns the server socket to an IP address and port (hanging the tin can on a specific hook on the wall). `listen()` announces that the server is ready to accept incoming connections.
  * **Accepting Connections:** When a client connects, `accept()` creates a **new socket specifically for that client**. This frees up the main server socket so other clients can still connect to the same address.
  * **Client Side:** The client creates its own socket and calls `connect((host, port))` to reach the server.
  * **Sending & Receiving:** Data is sent via `send()` and received via `recv()`. All messages must be encoded into bytes before sending and decoded back into strings when received.

* **Current Implementation & Bugs:**
  * Created a small chatroom script with `server.py` and `client.py` inside a new `network/` directory.
  * **Problem 1 (Max Clients):** The room currently crashes if a third client tries to join—it can only handle two clients.
  * **Problem 2 (Blocking I/O Bug):** If Client 1 joins and doesn't send a message, but Client 2 sends a message, the whole server freezes until Client 1 sends something. If Client 1 sends first, everything works fine.
  
* **Side Note:**
  * The head-to-head benchmark against PyTorch is still planned! I am just taking a short, fun detour into socket programming right now so I can eventually use it to build distributed training features for my tensor engine.

### 📌 September 2, 2026 — Head-to-Head PyTorch Parity Benchmark (Initial Accuracy Test)

Today I ran the first head-to-head benchmark comparing my custom tensor engine directly against PyTorch on the MNIST dataset to test for loss and accuracy parity.

* **Benchmark Setup & Architecture:**
  * **PyTorch Model (`pytorch_baseline.py`):** Built a baseline model inheriting from PyTorch's `nn.Module` using the standard `784 -> 128 -> 64 -> 10` structure with ReLU activations and Cross-Entropy Loss.
  * **Custom Model (`mymodel.py`):** Set up my custom engine model with the exact same architecture.
  * **Standardized Training:** Both models used a clean `train_model()` helper function to run 10 epochs with a batch size of 64 inside `pytorch_parity_benchmark.py`.

* **Debugging & Problem Solving:**
  * During the initial run, I hit a severe error saying gradients couldn't be calculated for my model.
  * AI suggested refactoring my core engine, but I knew my engine was solid from past tests and refused to rewrite it.
  * After auditing my new script, I found the bug: inside `mymodel.py`, I forgot to wrap `x_train` in my `Tensor` class! I wrote raw `x_train` instead of `Tensor(x_train)`.
  * Once I wrapped the input in `Tensor()`, the autograd graph connected properly and everything worked smoothly.

* **Initial Benchmark Results (Accuracy Parity):**
  * **PyTorch Baseline:** Finished with **97.45% test accuracy**with an average epoch time around **1.9s–2.1s**.
  * **Custom Engine:** Finished with **97.26% test accuracy**with epoch times ranging from **2.2s to ~5.9s**..

* **Performance Context & Reflection:**
  * **Speed vs. Accuracy:** I already expected PyTorch to win on execution speed because under the hood it's written in C++ and optimized by hundreds of top engineers. My engine is pure Python and NumPy running on a single thread.
  * **The Real Victory:** In terms of accuracy and loss trajectory, my engine stood completely **head-to-head with PyTorch** (97.26% vs 97.45%). Reaching near-identical convergence on a real dataset proves that the autograd graph, tensor operations, parameter update mechanics, and matrix calculus in my custom engine are mathematically sound and rock solid!

* **Note:** This is just the first benchmark focused purely on accuracy parity. I haven't added the charts to the `README.md` yet—I'll update the main README once I finish comparing all the other dimensions (like execution speed and epoch timing).

**Next Step:** Expand the benchmark comparisons to cover performance/timing dimensions, then organize the results for the `README.md`.

### 📌 September 4, 2026 — 4-Way Optimizer Benchmark (Adam, SGD, SGDM, RMSProp)

Today I decided to expand my optimizer comparison into a full 4-way benchmark! I wanted to compare all four major optimizers I learned about in Stanford CS231n: **Adam**, **SGD**, **SGD with Momentum (SGDM)**, and **RMSProp**.

* **Extending the benchmark (`optimizer.py`):**
  * Added `SGDM` and `RMSProp` classes to `optimizer.py`.
  * Implementing the formulas was pretty smooth because I had good notes from CS231n on velocity tracking for Momentum and moving averages of squared gradients for RMSProp.

* **Hyperparameter Tuning Headache:**
  * Updating `experiments/compare_optimizers.py` to run all four models gave me a huge headache with initial hyperparameters!
  * Standard SGD stuck around 42.42% accuracy, while SGDM was completely stuck at ~11.35% across all 10 epochs. 
  * After spending an hour tuning learning rates, I finally dialled them in and got every optimizer to converge nicely:
    * **Adam ($\text{lr} = 0.003$):** Hit **97.26% test accuracy** (Loss: ~0.02).
    * **RMSProp ($\text{lr} = 0.001$):** Finished highest at **97.90% test accuracy** (Loss: 0.0234).
    * **SGD ($\text{lr} = 0.05$):** Jumped up to **97.07% test accuracy** once the learning rate was increased (Loss: 0.0891).
    * **SGD with Momentum ($\text{lr} = 0.0007$):** Settled at **90.39% test accuracy** (Loss: 0.3244).

* **Takeaway:**
  * Seeing all four loss trajectories side-by-side shows how sensitive standard momentum and SGD are to exact learning rates compared to adaptive learning rate methods like Adam and RMSProp, which drop loss rapidly right from epoch 1.
  * *Note:* Full loss curves and exact metrics will be documented directly in the project `README.md`.

**Next Step:** Add these benchmark comparisons to the repository documentation and get back to working on distributed networking features!

### 📌 September 5, 2026 — Parameter Parity Benchmark across 5 Seeds, Repository Restructuring, & Tensor Socket Serialization

Today was packed with major performance validations, structural cleanup, and progress on the networking side.

* **Multi-Seed Benchmark with Weight Parity (`equalize_parameters`):**
  * **Methodology:** Created an `equalize_parameters()` helper to sync weights and biases between my engine and PyTorch before training, ensuring a completely fair starting point.
  * **Test Setup:** Evaluated both models across 5 distinct random seeds (`[22, 32, 42, 52, 62]`).
  * **Loss Parity:** The training loss curves for both models were literally overlaid on top of each other, confirming total mathematical and backprop equivalence.
  * **Accuracy:** My engine actually edged out PyTorch on 4 out of 5 seeds (`32`, `42`, `52`, `62`) and only lost by a fraction of a percent (about a tenth of a percent) on seed `22`.
  * **Execution Speed:** PyTorch ran consistently fast at 1.0s–2.5s per epoch. My engine averaged 3s–5s per epoch, with occasional spikes over 20 seconds.
  * *Note:* Graphs and comprehensive seed logs will be added to the main `README.md` in the coming days.

* **Repository Restructuring (`src/` Architecture):**
  * **Legacy Cleanup:** Moved the original 1D scalar engine from `micrograd/` into a root-level `legacy/` folder.
  * **Flat Directory Layout:** Moved `experiments/` to the root directory, and created a top-level `src/` directory containing `micrograd/` and `network/` to eliminate messy nested paths.
  * **Package Setup (`pyproject.toml`):** Added a `pyproject.toml` configuration to make the project an editable local package. Now I can import directly via `from micrograd.multi_dim_engine import Tensor` without hacking `sys.path` or `os` paths in my scripts!

* **Distributed Networking — Tensor Serialization:**
  * Successfully transmitted a live `Tensor` across the local network using Python's `pickle` library!
  * **Method:** Serialized `tensor.data` (the raw NumPy array) on the server, transmitted the bytes via TCP socket, and re-wrapped the received array into a `Tensor` object on the client side.


### 📌 September 6, 2026 — Researching Pickle Security Vulnerabilities vs. JSON

After research, I have found that using `pickle` is considered unsafe because unpickling doesn't just read data—it can execute code. 

* **The Problem with `pickle`:**
  * When you call `pickle.loads()` on something, you are not just decoding bytes into a Python object the way you can with JSON.
  * `pickle`'s format can include instructions that tell Python to actually execute arbitrary commands during the loading process.
  * If someone sends you a maliciously crafted `pickle` payload, calling `pickle.loads()` on your side can run code on your machine without you ever noticing. `pickle` doesn't have that safety boundary.

* **Why JSON / Safe Alternatives are Better:**
  * `json.loads()` can only ever produce plain data (strings, numbers, lists, dicts).
  * There is no way for a JSON payload to make your program do something beyond handling your data.

* **Takeaway:**
  * For network tensor serialization, I need to move away from `pickle` and use safe data formats (like raw byte buffers or JSON arrays) so receiving nodes aren't vulnerable to code injection.
