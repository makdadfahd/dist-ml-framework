import socket 
import numpy as np
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection
from network.rpcserver import RPCServer
import threading

#implement the constants :
server_address = socket.gethostbyname(socket.gethostname())
port = 5050

#initialise the server socket, bind it to (ADDRESS, PORT) and make the server waiting for connection :
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_address, port))
server_socket.listen()


#receiving the clients connection
while True :
    workersocket , address = server_socket.accept()
    print(f"Connection received from {address}\n")
    master = RPCServer(workersocket)
    #opening threads so we can handle multiple clients
    t = threading.Thread(target= master.handle_client)
    t.start()