from micrograd.multi_dim_engine import Tensor
from micrograd.optimizer import Adam
import numpy as np

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
