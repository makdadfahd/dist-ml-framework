import socket 
from network.rpcserver_sync import RPCServer
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

counter = [0]
received_grads = []
lock = threading.Lock()
condition = threading.Condition(lock)
event = threading.Event()


while True :
    clientsocket , addr = serversocket.accept()
    print("Connection received from a client")
    master = RPCServer(clientsocket, params, optimizer, condition, counter, received_grads)
    T1 = threading.Thread(target=master.handle_client)
    T1.start()

