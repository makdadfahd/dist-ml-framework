from micrograd.multi_dim_engine import Tensor
from micrograd.neural_network import MLP, cross_entropy_loss
from micrograd.optimizer import Adam , SGD ,SGDM , RMSProp
import numpy as np
import matplotlib.pyplot as plt
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

def run_experiment(epochs=10):
    np.random.seed(42)
    model_adam = MLP(784, [128, 64, 10])
    
    np.random.seed(42)
    model_sgd = MLP(784, [128, 64, 10])

    np.random.seed(42)
    model_sgdm = MLP(784, [128, 64, 10])

    np.random.seed(42)
    model_rmsprop = MLP(784, [128, 64, 10])

    optimizer_adam = Adam(model_adam.parameters())
    optimizer_sgd = SGD(model_sgd.parameters())
    optimizer_sgdm = SGDM(model_sgdm.parameters())
    optimizer_rmsprop = RMSProp(model_rmsprop.parameters())


    experiments = [
        ("Adam", model_adam, optimizer_adam),
        ("SGD", model_sgd, optimizer_sgd),
        ("SGDM", model_sgdm , optimizer_sgdm),
        ("RMSProp", model_rmsprop , optimizer_rmsprop)
    ]

    history = {
        "Adam": {"loss": [], "acc": []},
        "SGD": {"loss": [], "acc": []},
        "SGDM" : {"loss": [], "acc": []},
        "RMSProp" : {"loss": [], "acc": []}
    }

    num_batches = num_samples // batch_size

    for name, model, optimizer in experiments:
        print(f"\n--- Training {name} ---")
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            running_loss = 0.0

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
            history[name]["loss"].append(epoch_loss)

            test_scores = model(Tensor(X_test))
            test_preds = np.argmax(test_scores.data, axis=1)
            epoch_acc = np.mean(test_preds == y_test) * 100
            history[name]["acc"].append(epoch_acc)

            print(f"[{name}] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    return history


def plot_benchmark(history):
    epochs = list(range(1, len(history["Adam"]["loss"]) + 1))
    final_epoch = epochs[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 1. LOSS PLOT ---
    ax1.plot(epochs, history["Adam"]["loss"], label="Adam (lr=0.003)", color="#d3740e", linewidth=2)
    ax1.plot(epochs, history["SGD"]["loss"], label="SGD (lr=0.05)", color="#36ff0e", linewidth=2)
    ax1.plot(epochs, history["SGDM"]["loss"], label="SGD with Momentum (lr=0.0007)", color="#0e87ff", linewidth=2)
    ax1.plot(epochs, history["RMSProp"]["loss"], label="RMSProp (lr=0.001)", color="#ff0e0e", linewidth=2)


    # Annotate Final Loss Flags
    for name, color in [("Adam", "#d3740e"), ("SGD", "#36ff0e") , ("SGDM", "#0e87ff") , ("RMSProp", "#ff0e0e")]:
        final_loss = history[name]["loss"][-1]
        # Checkpoint dot
        ax1.scatter(final_epoch, final_loss, color=color, s=50, zorder=5)
        # Value Flag
        ax1.annotate(
            f"{final_loss:.4f}",
            (final_epoch, final_loss),
            textcoords="offset points",
            xytext=(8, -3),
            ha='left',
            fontweight='bold',
            color=color
        )

    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # --- 2. ACCURACY PLOT ---
    ax2.plot(epochs, history["Adam"]["acc"], label=f"Adam : {history["Adam"]["acc"][-1]:.2f}%", color="#d3740e", linewidth=2)
    ax2.plot(epochs, history["SGD"]["acc"], label=f"SGD : {history["SGD"]["acc"][-1]:.2f}%", color="#36ff0e", linewidth=2)
    ax2.plot(epochs, history["SGDM"]["acc"], label=f"SGD with Momentum : {history["SGDM"]["acc"][-1]:.2f}%", color="#0e87ff", linewidth=2)
    ax2.plot(epochs, history["RMSProp"]["acc"], label=f"RMSProp : {history["RMSProp"]["acc"][-1]:.2f}%", color="#ff0e0e", linewidth=2)
    
    # Annotate Final Accuracy Flags
    for name, color in [("Adam", "#d3740e"), ("SGD", "#36ff0e") , ("SGDM", "#0e87ff") , ("RMSProp", "#ff0e0e")]:
        final_acc = history[name]["acc"][-1]
        # Checkpoint dot
        ax2.scatter(final_epoch, final_acc, color=color, s=50, zorder=5)
        # Value Flag
        ax2.annotate(
            f"{final_acc:.2f}%",
            (final_epoch, final_acc),
            textcoords="offset points",
            xytext=(8, -3),
            ha='left',
            fontweight='bold',
            color=color
        )

    ax2.set_title("Test Accuracy Comparison (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Give extra right-margin space for the text flags so they don't clip off-screen
    ax1.set_xlim(1, final_epoch + 1.2)
    ax2.set_xlim(1, final_epoch + 1.2)
    
    plt.tight_layout()
    plt.savefig("optimizer_benchmark_flag.png", dpi=300)
    plt.show()

history = run_experiment(epochs=10)
plot_benchmark(history)