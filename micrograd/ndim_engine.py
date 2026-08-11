import numpy as np
import math as m

class Tensor : 
    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=float)
        self._prev = set(_children)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda : None 

    def __repr__(self):
        return f'Tensor(Data : {self.data})'

    def __add__(self, other) :
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other))

        def _backward() : 
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other)) 

        def _backward() : 
            pass
        out._backward = _backward
        
        return out

    def __matmul__(self, other):
        other =  other if isinstance(other,Tensor) else Tensor(other)

        self_is_1d = self.data.ndim == 1
        other_is_1d = other.data.ndim == 1

        self_matrix = self.data.reshape(1,-1) if self_is_1d else self.data
        other_matrix = other.data.reshape(-1,1) if other_is_1d else other.data

        out = self_matrix @ other_matrix

        if self_is_1d and other_is_1d :
            out = out.squeeze()
        elif self_is_1d :
            out = out.squeeze(axis = 0)
        elif other_is_1d :
            out = out.squeeze(axis = -1 )

        out = Tensor(out,(self,other))
        
        def _backward() :
            out_grad_matrix = out.grad
            if self_is_1d and other_is_1d :
                out_grad_matrix = out_grad_matrix.reshape(1,1)
            elif self_is_1d :
                out_grad_matrix = out_grad_matrix.reshape(1,-1)
            elif other_is_1d :
                out_grad_matrix = out_grad_matrix.reshape(-1 , 1)

            self_grad_matrix = out_grad_matrix @ other_matrix.T
            other_grad_matrix = self_matrix.T @ out_grad_matrix

            self.grad += self_grad_matrix.reshape(self.data.shape)
            other.grad += other_grad_matrix.reshape(other.data.shape)

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other 

    def __rmatmul__(self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other)
        return other @ self #not self @ other because @ is not commutative

    def __pow__(self, other):
        assert isinstance(other,(int,float))
        out = Tensor(self.data**other, (self,))

        def _backward() :
            pass
        out._backward = _backward

        return out

    def relu(self) : 
        out = Tensor([ max(0,xi) for xi in self.data ],(self,)) 
        def _backward() : 
            pass
        out._backward = _backward

        return out


    def backward(self) :

        topo = []
        visited = set()
        def topo_sort(node) :
            if node not in visited :
                visited.add(node)
                for child in node._prev :
                    topo_sort(child)
                topo.append(node)
        topo_sort(self)
        self.grad = 1.0

        for node in reversed(topo) :
            node._backward()

    def __neg__(self) :
        return self * (-1)

    def __sub__(self,other) :
        return self + (- other )

    def __rsub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * (other**(-1))

    def __rtruediv__(self, other):
        return other * (self**(-1))
    


x = Tensor([2,-1,2])
            

w = Tensor([[0,3,1],
            [2,0,1],
            [0,0,1]])

b = Tensor([1,1,1])

out = b + b

out.grad = np.ones(out.data.shape)
print(out.grad)
out._backward()

print(b.grad)
