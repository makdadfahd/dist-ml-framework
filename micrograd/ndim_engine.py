import numpy as np
import math as m

class Tensor : 
    def __init__(self, data, _children=()):
        self.data = np.array(data)
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda : None 

    def __repr__(self):
        return f'Tensor(Data : {self.data})'

    def __add__(self, other) :
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other))

        def _backward() : 
            pass
        out._backward = _backward

        return out

    def __mul__(self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other)) 

        def _backward() : 
            pass
        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self * other 

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
y = Tensor([4,2,1])

z = x.relu()

print(z,z._prev)