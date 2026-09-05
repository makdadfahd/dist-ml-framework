from micrograd.neural_network import MLP , cross_entropy_loss
from micrograd.optimizer import Adam
import numpy as np
from micrograd.multi_dim_engine import Tensor
import time 


class Engine_Model(MLP) :
    def __init__(self, n_inputs = 784 ,n_outputs = [128 , 64 , 10] , activation = 'relu') :
        super().__init__(n_inputs, n_outputs, activation=activation)

def train_model(model ,x_train , y_train , x_test , y_test , epochs = 10, lr=0.003, batch_size=64) :
    optimizer = Adam(model.parameters(), lr = lr)

    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size

    history = {"loss" : [] , "acc" : []}
    print("Starting Training Run... (Our Custom Model)\n" + "-" * 50)

    for epoch in range(epochs) :
        start_time = time.perf_counter()

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        running_loss = 0.0

        for i in range(num_batches) :
            idx = indices[i * batch_size : (i+1) * batch_size]
            x_batch , y_batch = Tensor(x_train[idx]) , y_train[idx]

            optimizer.zero_grad()
            scores = model(x_batch)
            loss = cross_entropy_loss(y_batch , scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data

        epoch_loss = running_loss / num_batches
        history['loss'].append(epoch_loss)

        epoch_scores = model(Tensor(x_test))
        epoch_pred = np.argmax(epoch_scores.data, axis = 1)
        epoch_acc = np.mean(epoch_pred == y_test) * 100
        history['acc'].append(epoch_acc)


        elapsed = time.perf_counter() - start_time
        print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
              f"Loss: {epoch_loss:.4f} | "
              f"Test Acc: {epoch_acc:.2f}% | "
              f"Time: {elapsed:.2f}s")

    print("-" * 50 + "\nTraining Complete.")

    return history