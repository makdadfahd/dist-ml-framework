import socket 
from network.rpcserver import RPCServer
from micrograd.neural_network import MLP
from micrograd.optimizer import Adam
from pytorch_comparaison.mymodel import Engine_Model
import threading

address = socket.gethostbyname(socket.gethostname())

#intialise the server socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind((address, 5050))
serversocket.listen()

#receive the client connection
clientsocket , addr = serversocket.accept()

model = Engine_Model()
params = [] 

for param in model.parameters() :
    params.append(param)

optimizer = Adam(params)

master = RPCServer(clientsocket, params, optimizer)

T1 = threading.Thread(target=master.handle_client)
T1.start()
T1.join()

for param in model.parameters() :
    print(param.grad)