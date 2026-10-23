### 📌 October 5, 2026 — Testing `push_grads`, Fixing RPC `*args` Unpacking Bug, & Validating End-to-End Weight/Grad Transfer

After a brief break due to the start of the new university semester, I jumped back into the project to test the end-to-end gradient transmission pipeline via RPC.

* **Client-Side Forward/Backward Test Pipeline:**
  * **Weight Synchronization:** Called `get_weights()` via RPC, looped over the local model parameters, and initialized each client layer with the exact weights received from the server.
  * **Forward & Backward Pass:** Loaded the MNIST dataset, ran a forward pass on 2 sample inputs, calculated the loss, and executed `loss.backward()` to compute local gradients.
  * **Gradient Extraction:** Iterated through the parameter graph, appended each tensor's `.grad` attribute to a `gradients` list, and dispatched a `call("push_grads", gradients)` RPC request.

* **Debugging the Unpacking Bug (`TypeError` / `RuntimeError`):**
  * **The Error:** Running the client triggered a `RuntimeError` stating that `RPCServer.push_grads()` expected 2 positional arguments but received 7.
  * **Isolating the Cause:** Checked the length of `gradients`, which was exactly 6. Adding 1 for the function name yielded 7 arguments, confirming that the list elements were being unpacked individually as separate function arguments instead of being passed as a single list object.
  * **Root Cause Location:** While `serve_request()` was clean, the bug originated in `dispatch()`: I had originally written `func(*args)` to handle simple variadic functions (like `add(1, 2, 3, 4)`). The asterisk `*` explicitly unpacked the array items into separate positional arguments.

* **Resolution & Conditional Execution:**
  * Removing `*` directly broke `get_weights()` and standard calls that depended on argument unpacking.
  * Implemented conditional invocation logic inside `dispatch()`:
    ```python
    result = func(*args) if function_name != "push_grads" else func(args)
    ```
  * With this conditional handling in place, both `get_weights()` and `push_grads()` executed without errors!

* **Outcome:**
  * Successfully initialized model parameters remotely on worker nodes and transmitted computed parameter gradients back to the server.

### 📌 October 8, 2026 — Refactoring RPC Argument Unpacking in `dispatch()` & Cleaning Up `push_grads` Invocation

After stepping back and reflecting on the hardcoded `if` statement in `dispatch()`, I realized it was only a temporary, brittle fix. Hardcoding specific method names like `push_grads` would force me to update the conditional check every time a new method expecting a list argument was added to the framework.

* **Refactoring `dispatch()`:**
  * Removed the hardcoded `if function_name != "push_grads"` statement entirely from `rpcserver.py`.
  * Restored the clean, universal `result = func(*args)` execution line in `dispatch()`.

* **Elegant Client-Side Solution:**
  * **The Root Cause:** When calling `self.call("push_grads", gradients)` where `gradients` is `[g1, g2, ...]`, `*args` unpacked `gradients` into individual positional arguments `g1 , g2 , ...`.
  * **The Fix:** Wrapped `gradients` inside an outer list on the client side: `self.call("push_grads", [gradients])`.
  * **How it Works:** When `*args` unpacks `[gradients]`, it extracts the outer list wrapper, correctly passing the single underlying `gradients` list as the sole positional argument to `push_grads(self, grads)`.

* **Takeaway:**
  * Solving the issue at the caller interface kept the core server dispatcher generic, clean, and extensible for future RPC methods without introducing hardcoded execution exceptions.

### 📌 October 19–22, 2026 — Distributed Training on MNIST, Gradient Bug Fix, & Making Epochs Faster

After getting `get_weights` and `push_grads` working, I decided to test real training on the MNIST dataset using a master node and a worker node.

* **First Distributed Test:**
  * I loaded the dataset on the worker side. For every single batch, the worker called `get_weights`, ran forward and backward passes, and sent gradients back with `push_grads`.
  * **The Problem:** It was super slow! Each epoch took **5 to 6 minutes** (instead of 3.84s locally), and accuracy was terrible—starting at **10.10%** and dropping to **8.64%** in epoch 2.

* **Bug #1 — Stale Gradients Fix:**
  * **The Bug:** I thought zeroing gradients on the server side would also zero them on the client side. But because they are separate objects over the network, the client kept accumulating old gradients. So when `.backward()` ran, the gradients were totally wrong.
  * **The Fix:** I added `zero_grad()` directly on the client side before every batch.
  * **The Result:** Accuracy jumped back to normal! It started at **90%** in epoch 1 and reached over **97%** by epoch 10, matching local training.

* **Bug #2 — Network Speed Bottleneck:**
  * Even though accuracy was good, epochs were still taking **6 to 10 minutes**. Calling `get_weights` and `push_grads` for every single batch (over 900+ calls per epoch) was choking the network connection.

* **Optimization — Syncing Every 10 Batches:**
  * **The Solution:** Instead of sending weights and gradients on every batch, I changed the code to sync only once every 10 batches.
  * **The Result:** Time per epoch dropped down to just **9 to 12 seconds**!
  * **Trade-off:** Accuracy dropped a bit—starting at **50%** in epoch 1 and reaching **93.69%** in epoch 10, because the worker used slightly old weights between syncs.


### 📌 October 23, 2026 — Solving Network Latency: Tensor Packing, FP16 Compression, & Disabling Nagle’s Algorithm (`TCP_NODELAY`)

Today was a huge breakthrough day for the distributed system's training performance!

* **Client-Side Optimizer Attempt:**
  * Tried setting up a local Adam optimizer on the client side to boost accuracy while keeping the 10-batch sync stride.
  * **Result:** Accuracy only went up by ~0.2% (reaching **93.88%**), so this wasn't enough to reach parity with local training.

* **Returning to Per-Batch Syncing:**
  * Switched back to calling `get_weights` and `push_grads` every batch to restore full accuracy (**97%+** by epoch 10).
  * **The Goal:** Fix the massive **6 to 10 minute** per-epoch bottleneck while keeping per-batch weight synchronization intact.

* **Optimization Attempt 1 — Bulk Tensor Packing:**
  * Instead of sending tensors individually in a loop, I modified the pipeline to send tensors as massive unified blocks over the socket.
  * **Result:** Epoch time dropped significantly from **6–10 minutes down to ~90 seconds**, while accuracy stayed high at **97%**.

* **Optimization Attempt 2 — Precision Compression (FP32 $\rightarrow$ FP16):**
  * Cast tensors from `float32` down to `float16` before sending over the network, then converted them back to `float32` upon receipt to cut the amount of data sent in half.
  * **Result:** Time dropped slightly to **~85 seconds** per epoch, proving the amount of data sent wasn't the main bottleneck.

* **The Game Changer — Disabling Nagle’s Algorithm (`TCP_NODELAY`):**
  * **The Insight:** Realized the latency was caused by Nagle’s algorithm, which buffers small TCP packets by default to save network bandwidth. While good for web browsing, it introduces terrible latency for small, rapid RPC requests in real-time distributed learning.
  * **The Fix:** Enabled low-latency transmission on the socket with a single line:
    ```python
    clientsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ```
  * **The Result:** Time per epoch dropped from **85s down to an average of just 8.14s**—a massive speedup while maintaining full **97%+** accuracy!