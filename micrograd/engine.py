import numpy as np
import math as m

class Value : 
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda : None 

    def __repr__(self):
        return f'Value(Data : {self.data})'

    def __add__(self, other) :
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other))

        def _backward() : 
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self,other) :
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other)) 

        def _backward() : 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self * other 

    def __pow__(self, other):
        assert isinstance(other,(int,float)) , "only supporting int/float powers for now"
        out = Value(self.data**other, (self,))

        def _backward() :
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward

        return out

    def tanh(self) : 
        x = self.data 
        t = (m.exp(x)**2 -  1) / (m.exp(x)**2 + 1)
        out = Value(t, (self,))

        def _backward() :
            self.grad = (1 - t**2) * out.grad 
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


