import sys , os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


#start by importing the necessary libraries
from mymodel import Engine_Model , train_model
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torchvision import datasets
from multi_dim_engine import Tensor

#loading the datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)


#organizing it in numpy arrays
x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_test = test_dataset.targets.numpy()

np.random.seed(42)

#initialising the models 
my_model = Engine_Model()

_ = train_model(my_model, x_train , y_train, x_test, y_test)


#plotting the images
batch_size = 28
indices = np.random.choice(len(x_test), size=batch_size, replace=False)

x_sample = Tensor(x_test[indices])
y_true = y_test[indices]

scores = my_model(x_sample)
predictions = np.argmax(scores.data, axis=1)

# 1. Compact figure size matching grid ratio (7:4)
fig, axes = plt.subplots(4, 7, figsize=(14, 8))

for i, ax in enumerate(axes.flat):
    image = x_test[indices[i]].reshape(28, 28)
    ax.imshow(image, cmap="gray")
    
    pred = predictions[i]
    true = y_true[i]
    
    color = "green" if pred == true else "red"
    # Tight title positioning directly above images
    ax.set_title(f"{pred} ({true})", color=color, fontsize=10, pad=2)
    ax.axis("off")

# 2. Control subplots spacing directly (wspace=horizontal gap, hspace=vertical gap)
plt.subplots_adjust(wspace=0.02, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.01)

plt.savefig("mnist_benchmark.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.show()