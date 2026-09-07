import socket 
import numpy as np
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection


#implement the constants :
server_address = socket.gethostbyname(socket.gethostname())
port = 5050
test_tensor = Tensor([[1 , 2 , 3],
                 [4 , 5 , 6]     ,
                 [7 , 8 , 9]])


#initialise the server socket, bind it to (ADDRESS, PORT) and make the server waiting for connection :
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_address, port))
server_socket.listen()


#receiving the clients connection
clientsocket , address = server_socket.accept()
print(f"Connection received from {address}\n")


#sending tensor to the worker

master = TensorConnection(clientsocket)
master.send_tensor(test_tensor)
print("Tensor sent successfully...")


#closing client and server socket
clientsocket.close()
server_socket.close()