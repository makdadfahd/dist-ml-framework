import sys , os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


#start by importing the necessary libraries
from mymodel import Engine_Model , train_model
from pytorch_baseline import Model_pytorch , train_pytorch
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torchvision import datasets
import random

#loading the datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)


#organizing it in numpy arrays
x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_test = test_dataset.targets.numpy()


#running the benchmark, we setup the seeds first so both can have equal start point
#torch.manual_seed(42) and np.random.seed(42) can't guarantee the same intial parameters, so i need a function that will ensure that

def set_seed(torch_params, my_params):
    for torch_param , my_param in zip(torch_params, my_params) :
        my_param.data[:] = torch_param.detach().numpy()


#defining the seeds 
seeds = [22 , 32 , 42 , 52, 62]


#looping trough seeds

for seed in seeds :
    print(f"Benchmarks starts on seed : {seed}\n")
    #initialising my model
    my_model = Engine_Model()


    #initialising pytorch model
    torch.manual_seed(seed)

    pytorch_model = Model_pytorch()

    set_seed(pytorch_model.parameters(), my_model.parameters())

    #training them on MNIST dataset and getting their history of loss and accuracy
    pytorch_history = train_pytorch(pytorch_model, x_train , y_train, x_test, y_test)
    mymodel_history = train_model(my_model, x_train , y_train, x_test, y_test)


    #setting the graph parameters 
    epochs = range(1,11)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, mymodel_history["loss"], label="Custom Adam", color="#1f77b4")
    plt.plot(epochs, pytorch_history["loss"], label="PyTorch Adam", color="#EE4C2C", linestyle="--")
    plt.title("Loss Trajectory")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mymodel_history["acc"], label="Custom Adam", color="#1f77b4")
    plt.plot(epochs, pytorch_history["acc"], label="PyTorch Adam", color="#EE4C2C", linestyle="--")
    plt.title("Test Accuracy (%)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"pytorch_vs_custom_{seed}.png", dpi=300)