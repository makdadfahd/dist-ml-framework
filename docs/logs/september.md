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

