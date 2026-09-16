import socket 
import numpy as np
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection
from network.rpcserver import RPCServer


#implement the constants :
dest_address = socket.gethostbyname(socket.gethostname())
dest_port = 5050


#intialize the client socket, and connect to the server :
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_address, dest_port))


#receiving the Tensor from master
worker = RPCServer(client_socket)

print("We will ask the server what is 2 + 3 ?")
print("Waiting for response...")
response = worker.call("add", [2,3,4,5,6])
print(f'The answer is : {response}')

print("\nResponse received successfully...")
#closing the client socket after receiving the array
client_socket.close()