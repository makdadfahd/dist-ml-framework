import numpy as np 

class Tensor :
    def __init__(self, data = np.array([]), _children = [] ):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"Tensor( Data : {self.data} )"

    def __add__(self, other):
        out = self.data + other.data
        return Tensor(out, (self,other))
    
    def __sub__(self,other) :
        out = self.data - other.data
        return Tensor(out, (self,other))
    
    def __mul__(self, other):
        out = self.data * other.data
        return Tensor(out , (self,other))
    
    def __truediv__(self, other):
        eps = 1e-8
        out = self.data / (other.data + eps)
        return Tensor(out, (self,other))

    def __pow__(self, other):
        if isinstance(other, Tensor) :
            exponement = other.data
        else :
            exponement = other
        out = self.data**exponement
        
        return Tensor(out,(self,))

    def __matmul__(self, other):
        out = self.data @ other.data
        return Tensor(out,(self,other))

    def __rmatmul__(self, other):
        return self.data @ other.data



