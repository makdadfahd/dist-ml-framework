import numpy as np

class Value : 
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda : None 

    def __repr__(self):
        return f'Data : {self.data}'

    def __add__(self, other) :
        out = Value(self.data + other.data, (self,other))

        def _backward() : 
            self.grad = 1.0 * out.grad 
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self,other) :
        out = Value(self.data * other.data, (self,other)) 

        def _backward() : 
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        
        return out



a = Value(2.0)
b = Value(3.0)
c = Value(9.3)
d = a + b
e = d * c 
f = Value(0.5)
g = e * f 

g.grad = 1.0

g._backward()
e._backward()
print(d.grad)