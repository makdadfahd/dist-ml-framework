import socket
from network.rpcserver import RPCClient
from micrograd.multi_dim_engine import Tensor
from micrograd.neural_network import MLP , cross_entropy_loss
from torchvision import datasets
from pytorch_comparaison.mymodel import Engine_Model

address = socket.gethostbyname(socket.gethostname())

#initialise the client socket and connect to the server
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((address, 5050))

worker = RPCClient(clientsocket)

#intialising the MLP
localModel = Engine_Model()

weights = worker.call("get_weights")
parameters = localModel.parameters()

for i , param in enumerate(parameters) :
    param.data = weights[i].data

#importing a single MNIST image 
train_dataset = datasets.MNIST(root= './data', train= True, download = True)

x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

input = Tensor(x_train[0:2])
output = y_train[0:2]

scores = localModel(input)

loss = cross_entropy_loss(output, scores)

loss.backward()

gradients = []

for param in parameters :
    gradients.append(Tensor(param.grad))

message = worker.call("push_grads", [gradients])

print(message)