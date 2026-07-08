import numpy as np 

class Tensor :
    def __init__(self, data = None):
        if data is None :
            self.data = np.array([])
        else :
            self.data = np.array(data)

    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __sub__(self,other) :
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data)
    
    def __truediv__(self, other):
        eps = 1e-8
        return Tensor(self.data / (other.data + eps))

    def __pow__(self, other):
        if isinstance(other, Tensor) :
            exponement = other.data
        else :
            exponement = other
        return Tensor(self.data**exponement)

