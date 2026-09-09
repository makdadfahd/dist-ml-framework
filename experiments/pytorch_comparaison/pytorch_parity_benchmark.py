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
seeds = [0 , 1 , 2 , 3, 4]
all_custom_losses = []
all_pytorch_losses = []
all_custom_acc = []
all_pytorch_acc= []
 
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

    #appending the accuracy and loss we have got to the all loss and accuracy lists to make a 2D array
    all_custom_losses.append(mymodel_history["loss"])
    all_pytorch_losses.append(pytorch_history["loss"])
    all_custom_acc.append(mymodel_history["acc"])
    all_pytorch_acc.append(pytorch_history["acc"])

#getting the mean and std
all_custom_losses = np.array(all_custom_losses)
mean_custom_loss = all_custom_losses.mean(axis = 0)
std_custom_loss = all_custom_losses.std(axis = 0)

all_pytorch_losses = np.array(all_pytorch_losses)
mean_pytorch_loss = all_pytorch_losses.mean(axis = 0)
std_pytorch_loss = all_pytorch_losses.std(axis = 0)

all_custom_acc = np.array(all_custom_acc)
mean_custom_acc = all_custom_acc.mean(axis = 0)
std_custom_acc = all_custom_acc.std(axis = 0)

all_pytorch_acc = np.array(all_pytorch_acc)
mean_pytorch_acc = all_pytorch_acc.mean(axis = 0)
std_pytorch_acc = all_pytorch_acc.std(axis = 0)

#plotting the results :
epochs = range(1,11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, mean_custom_loss, label="Custom Adam")
ax1.fill_between(epochs, mean_custom_loss - std_custom_loss, mean_custom_loss + std_custom_loss, alpha=0.2)
ax1.plot(epochs, mean_pytorch_loss, label="PyTorch Adam")
ax1.fill_between(epochs, mean_pytorch_loss - std_pytorch_loss, mean_pytorch_loss + std_pytorch_loss, alpha=0.2)
ax1.set_title("Loss Trajectory (5 Seeds)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs, mean_custom_acc, label="Custom Adam")
ax2.fill_between(epochs, mean_custom_acc - std_custom_acc, mean_custom_acc + std_custom_acc, alpha=0.2)
ax2.plot(epochs, mean_pytorch_acc, label="PyTorch Adam")
ax2.fill_between(epochs, mean_pytorch_acc - std_pytorch_acc, mean_pytorch_acc + std_pytorch_acc, alpha=0.2)
ax2.set_title("Test Accuracy (5 Seeds)")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.savefig("comparison_5seeds.png")