import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_dim_engine import Tensor
from nn import MLP, cross_entropy_loss
from optimizer import Adam
import numpy as np

import torch
from torchvision import datasets

train_dataset = datasets.MNIST(root='./data', train=True, download=True)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

batch_size = 64
num_samples = X_train.shape[0]

test_dataset = datasets.MNIST(root="./data", train=False, download=True)

X_test = test_dataset.data.numpy().reshape(-1, 784).astype("float32") / 255.0
y_test = test_dataset.targets.numpy()


import matplotlib.pyplot as plt

def run_experiment(activation_name, epochs=10, lr=0.001):
    np.random.seed(42)
    model = MLP(784, [128, 64, 10], activation=activation_name)
    optimizer = Adam(model.parameters(), lr)
    
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        running_loss = 0.0
        num_batches = num_samples // batch_size

        for i in range(num_batches):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            x_batch = Tensor(X_train[idx])
            y_batch = y_train[idx]

            scores = model(x_batch)
            loss = cross_entropy_loss(y_batch, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data

        epoch_loss = running_loss / num_batches
        loss_history.append(epoch_loss)

        test_scores = model(Tensor(X_test))
        test_preds = np.argmax(test_scores.data, axis=1)
        epoch_acc = np.mean(test_preds == y_test) * 100
        acc_history.append(epoch_acc)

        print(f"[{activation_name}] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    return loss_history, acc_history

activations = ['relu', 'tanh', 'sigmoid']
history = {}

for act in activations:
    print(f"\n--- Running {act.upper()} ---")
    loss_hist, acc_hist = run_experiment(act, epochs=10, lr=0.001)
    history[act] = {'loss': loss_hist, 'acc': acc_hist}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
epochs_range = range(1, 11)

for act, metrics in history.items():
    ax1.plot(epochs_range, metrics['loss'], marker='o', label=act.upper())
    ax2.plot(epochs_range, metrics['acc'], marker='o', label=f"{act.upper()} ({metrics['acc'][-1]:.2f}%)")

ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Average Batch Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.set_title("Test Accuracy per Epoch (%)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("activation_benchmark.png", bbox_inches="tight")
print("\nSaved real benchmark plot to activation_benchmark.png!")


