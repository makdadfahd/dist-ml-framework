import socket 
import numpy as np
import pickle
from micrograd.multi_dim_engine import Tensor

#implement the constants :
dest_address = socket.gethostbyname(socket.gethostname())
dest_port = 5050


#intialize the client socket, and connect to the server :
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_address, dest_port))


#receiving the array data
data = client_socket.recv(4096)
tensor = Tensor(pickle.loads(data))
print(f"{tensor}")


#closing the client socket after receiving the array
client_socket.close()