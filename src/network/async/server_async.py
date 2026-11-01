import socket 
from network.rpcserver_async import RPCServer
from micrograd.neural_network import MLP
from micrograd.optimizer import Adam
from pytorch_comparaison.mymodel import Engine_Model
import threading

address = socket.gethostbyname(socket.gethostname())

#intialise the server socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

serversocket.bind((address, 5050))
serversocket.listen()

#receive the client connection

model = Engine_Model()
params = [] 

for param in model.parameters() :
    params.append(param)

optimizer = Adam(params)

client_counter = 0


while True :
    clientsocket , addr = serversocket.accept()
    client_counter += 1
    print(f"Connection received from a client {client_counter}")
    master = RPCServer(clientsocket, params, optimizer)
    T1 = threading.Thread(target=master.handle_client)
    T1.start()

