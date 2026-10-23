import socket, time
from network.rpcserver import RPCClient
from micrograd.multi_dim_engine import Tensor
from micrograd.neural_network import MLP , cross_entropy_loss
from torchvision import datasets
from pytorch_comparaison.mymodel import Engine_Model
import numpy as np
from micrograd.optimizer import Adam



address = socket.gethostbyname(socket.gethostname())

#initialise the client socket and connect to the server
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
clientsocket.connect((address, 5050))

worker = RPCClient(clientsocket)

#intialising the MLP
localModel = Engine_Model()


parameters = localModel.parameters()


#importing a single MNIST image 
train_dataset = datasets.MNIST(root= './data', train= True, download = True)

x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

test_dataset = datasets.MNIST(root="./data", train=False, download=True)

x_test = test_dataset.data.numpy().reshape(-1, 784).astype("float32") / 255.0
y_test = test_dataset.targets.numpy()

#organising data
batch_size = 64
num_samples = x_train.shape[0]
epochs = 10

loss_history = []
acc_history = []

#running the training
for epoch in range(epochs):
    start_time = time.perf_counter()
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    running_loss = 0.0
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        idx = indices[i * batch_size : (i + 1) * batch_size]
        x_batch = Tensor(x_train[idx])
        y_batch = y_train[idx]

        weights = worker.call("get_weights")
        for index , param in enumerate(parameters) :
            param.data = weights[index].data
        
        scores = localModel(x_batch)
        loss = cross_entropy_loss(y_batch, scores)

        for param in parameters :
            param.grad = np.zeros_like(param.data)

        loss.backward()

        gradients = []

        for param in parameters :
            gradients.append(Tensor(param.grad))
        
        message = worker.call("push_grads", gradients)
            
        running_loss += loss.data   

    elapsed = time.perf_counter() - start_time    

    epoch_loss = running_loss / num_batches
    loss_history.append(epoch_loss)

    test_scores = localModel(Tensor(x_test))
    test_preds = np.argmax(test_scores.data, axis=1)
    epoch_acc = np.mean(test_preds == y_test) * 100
    acc_history.append(epoch_acc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Time: {elapsed:.2f}s")


