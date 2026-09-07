import socket 
import numpy as np
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection

#implement the constants :
dest_address = socket.gethostbyname(socket.gethostname())
dest_port = 5050


#intialize the client socket, and connect to the server :
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_address, dest_port))


#receiving the Tensor from master
worker = TensorConnection(client_socket)
test_tensor = worker.recv_tensor()
print(f"{test_tensor}")

print("\nTensor received successfully...")
#closing the client socket after receiving the array
client_socket.close()