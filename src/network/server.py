import socket , threading
from micrograd.multi_dim_engine import Tensor
from network.tensorconnection import TensorConnection
from network.rpcserver import RPCServer
from micrograd.optimizer import Adam
from torchvision import datasets
import numpy as np
from pytorch_comparaison.mymodel import Engine_Model

#loading the datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)


#organizing it in numpy arrays
x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_test = test_dataset.targets.numpy()

#implement the constants :
server_address = socket.gethostbyname(socket.gethostname())
port = 5050

#initialise the server socket, bind it to (ADDRESS, PORT) and make the server waiting for connection :
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_address, port))
server_socket.listen()


#get the parameters and initialise the optimizer
model = Engine_Model()
parameters = model.parameters()
optimizer = Adam(parameters, lr = 0.003)

#receiving the clients connection
while True :
    workersocket , address = server_socket.accept()
    print(f"Connection received from {address}\n")
    master = RPCServer(workersocket, parameters, optimizer)

    #opening threads so we can handle multiple clients
    t = threading.Thread(target= master.handle_client)
    t.start()