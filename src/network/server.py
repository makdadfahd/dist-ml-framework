import socket 
from network.rpcserver import RPCServer
from micrograd.neural_network import MLP
from micrograd.optimizer import Adam

address = socket.gethostbyname(socket.gethostname())

#intialise the server socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind((address, 5050))
serversocket.listen()

#receive the client connection
clientsocket , addr = serversocket.accept()

model = MLP(3, [2,2,1])
params = [] 

for param in model.parameters() :
    params.append(param)

optimizer = Adam(params)

master = RPCServer(clientsocket, params, optimizer)

while True :
    master.handle_client()