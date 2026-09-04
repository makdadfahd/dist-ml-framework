from multi_dim_engine import Tensor
import numpy as np

class Adam : 
    def __init__(self, params, lr = 0.003 , beta1 = 0.9 , beta2 = 0.99 , eps = 1e-8 ):
        self.params = params
        self.epsilon = eps 
        self.learning_rate = lr 
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) :
        self.t += 1 
        for i , p in enumerate(self.params) :
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2 

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.learning_rate * (m_hat/(np.sqrt(v_hat) + self.epsilon))

    def zero_grad(self) :
        for p in self.params :
            p.grad = np.zeros_like(p.data)


class SGD :
    def __init__(self, params , alpha = 0.05):
        self.alpha = alpha
        self.params = params

    def step(self) :
        for p in self.params :
             p.data -= self.alpha * p.grad

    def zero_grad(self) :
        for p in self.params :
            p.grad = np.zeros_like(p.data)


class SGDM :
    def __init__(self, params , rho = 0.9 , alpha = 0.0007) :
        self.params = params
        self.rho = rho
        self.alpha = alpha
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) :
        for i , p in enumerate(self.params) :
            self.v[i] = self.rho * self.v[i] + p.grad
            p.data -= self.alpha * self.v[i]

    def zero_grad(self) :
        for p in self.params :
            p.grad = np.zeros_like(p.data)


class RMSProp :
    def __init__(self , params , beta = 0.9 , eps = 1e-8 , lr = 0.001) :
        self.params = params
        self.beta = beta
        self.eps = eps
        self.learning_rate = lr
        self.v = [np.zeros_like(p.data) for p in self.params]


    def step(self) :
        for i , p in enumerate(self.params) :
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * p.grad**2 

            p.data -= (self.learning_rate / np.sqrt(self.v[i] + self.eps)) * p.grad

    def zero_grad(self) :
        for p in self.params :
            p.grad = np.zeros_like(p.data)
        
