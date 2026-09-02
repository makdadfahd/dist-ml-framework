import sys , os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import MLP , cross_entropy_loss
from optimizer import Adam
import numpy as np
from multi_dim_engine import Tensor

class Engine_Model(MLP) :
    def __init__(self, n_inputs = 784 ,n_outputs = [128 , 64 , 10] , activation = 'relu') :
        super().__init__(n_inputs, n_outputs, activation=activation)

    def train_model(model ,x_train , y_train , x_test , y_test , epochs = 10, lr=0.003, batch_size=64) :
        optimizer = Adam(model.parameters(), lr = lr)

        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size

        history = {"loss" : [] , "acc" : []}

        for epoch in range(epochs) :
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            running_loss = 0.0

            for i in range(num_batches) :
                idx = indices[i * batch_size : (i+1) * batch_size]
                x_batch , y_batch = x_train[idx] , y_train[idx]

                optimizer.zero_grad()
                scores = model(x_batch)
                loss = cross_entropy_loss(y_batch , scores)
                loss.backward()
                optimizer.step()

                running_loss += loss.data

            epoch_loss = running_loss / num_batches
            history['loss'].append(epoch_loss)

            epoch_scores = model(x_train)
            epoch_pred = np.argmax(epoch_scores.data, axis = 1)
            epoch_acc = np.mean(epoch_pred == y_train) * 100
            history['acc'].append(epoch_acc)

        return history