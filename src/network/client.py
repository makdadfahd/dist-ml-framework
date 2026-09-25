import socket
from network.rpcserver import RPCClient


address = socket.gethostbyname(socket.gethostname())

#initialise the client socket and connect to the server
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((address, 5050))

worker = RPCClient(clientsocket)

weights = worker.call("get_weights")

print(weights)