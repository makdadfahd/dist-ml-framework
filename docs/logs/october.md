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

