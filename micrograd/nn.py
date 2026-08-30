from multi_dim_engine import Tensor
from optimizer import Adam
import numpy as np
import torch
from torchvision import datasets

#loading the training dataset :

train_dataset = datasets.MNIST(root='./data', train=True, download=True)
X_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()

#loading the test dataset :

test_dataset = datasets.MNIST(root="./data", train=False, download=True)
X_test = test_dataset.data.numpy().reshape(-1, 784).astype("float32") / 255.0
y_test = test_dataset.targets.numpy()


def cross_entropy_loss(index_vector,scores) :
    probabilities = softmax(scores)

    rows = np.arange(probabilities.data.shape[0])
    true_prob = Tensor(probabilities.data[rows, index_vector])

    losses = Tensor(-np.log(true_prob.data))
    N = true_prob.data.shape[0]
    
    full_loss = Tensor(losses.data.mean() ,(scores,))

    def _backward() :
        num_classes = probabilities.data.shape[1]
        Y = np.eye(num_classes)[index_vector]
        scores.grad = (probabilities.data - Y)/N
        
    full_loss._backward = _backward
        
    return full_loss

def softmax(scores) :
    shifted_scores = scores.data - np.max(scores.data , axis = -1 , keepdims=True)
    expon = np.exp(shifted_scores)
    prob = expon / np.sum(expon, axis = -1, keepdims=True)
    prob = np.clip(prob, 1e-12 , 1.0 - 1e-12)
    return Tensor(prob)
    

class Layer :

    def __init__(self, n_inputs , n_neurons) :
        scale = np.sqrt(2.0 / n_inputs)
        self.w = Tensor(np.random.rand(n_neurons,n_inputs) * scale)
        self.b = Tensor(np.random.rand(n_neurons))

    def __call__(self,x) :
        output = x @ self.w.T() + self.b
        return output

    def parameters(self) :
        return [self.w , self.b]
    
    
class MLP : 

    def __init__(self, n_inputs,n_outputs, activation = 'relu') :
        size = [n_inputs] + n_outputs 
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)]
        self.activation = activation


    def __call__(self,x) :
        for layer in self.layers[:-1] : 
            x = layer(x)
            if self.activation == 'relu' :
                x = x.relu()
            elif self.activation == 'sigmoid' :
                x = x.sigmoid()
            elif self.activation == 'tanh' :
                x = x.tanh()
        scores = self.layers[-1](x)
        return scores

    def parameters(self) :
        params = []
        for layer in self.layers :
            params.extend(layer.parameters())
        return params

    def learn(self,batch_size, epochs = 20) :
        num_samples = X_train.shape[0]
        optimizer = Adam(self.parameters(), 0.003)  

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            running_loss = 0.0
            num_batches = num_samples // batch_size

            for i in range(num_batches):
                idx = indices[i * batch_size : (i + 1) * batch_size]
                x_batch = Tensor(X_train[idx])
                y_batch = y_train[idx]

                scores = self(x_batch)
                loss = cross_entropy_loss(y_batch, scores)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data
            print(
                    f"Epoch {epoch + 1} | Average Loss = {running_loss / num_batches:.4f}"
                )

        test_scores = self(Tensor(X_train))
        test_preds = np.argmax(test_scores.data, axis=1)

        accuracy = np.mean(test_preds == y_train) * 100
        print(f"Accuracy on training data : {accuracy:.2f}%")

    def pred(self, batch_size) :
        indices = np.random.choice(len(X_test), size=batch_size, replace=False)
        x_sample = Tensor(X_test[indices])
        y_true = y_test[indices]

        scores = self(x_sample)
        predictions = np.argmax(scores.data, axis=1)

        accuracy = np.mean(predictions == y_true) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

