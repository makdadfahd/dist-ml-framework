import socket 
import numpy as np
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection
from network.rpcserver import RPCServer , RPCClient


#implement the constants :
dest_address = socket.gethostbyname(socket.gethostname())
dest_port = 5050
A = Tensor([[1,2,3], [4,5,6], [7,8,9]])
B = Tensor([[1,0,0], [0,1,0], [0,0,1]])

#intialize the client socket, and connect to the server :
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_address, dest_port))


#receiving the Tensor from master
worker = RPCClient(client_socket)

response = worker.call("add", [2,B])
print(f'The answer is : {response}')

print("\nResponse received successfully...")
    
client_socket.close()