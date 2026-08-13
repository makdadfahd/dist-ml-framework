from engine import Value
import random

class Neuron :
    def __init__(self,n_input) :
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.b = Value(random.uniform(-1 , 1))

    def __call__(self,x) :
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        output = act.tanh()
        #didn't understand why i can't do act.engine.Value.tanh()
        return output

    def parameters(self) :
        return self.w + [self.b]

class Layer : 
    def __init__(self, n_inputs, n_outputs) :
        # the number of neurons is the same as the number of outputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x) :
        outputs = [neuron(x) for neuron in self.neurons ] 
        #started to master using for loops at the same line :)
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) :
        params = []
        for neuron in self.neurons :
            ps = neuron.parameters()
            params.extend(ps)
        return params
class MLP : 
    def __init__(self,n_inputs, n_outputs) :
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)]
        #fixed a bug of i out of range : by decreasing the length of the size by 1 

    def __call__(self, x) :
        for layer in self.layers :
            x = layer(x)
        return x 
    
    def parameters(self) :
        params = []
        for layer in self.layers :
            ps = layer.parameters()
            params.extend(ps)
        return params
    
    def get_result(self,iterations,alpha,xs,ys) :
        for i in range(iterations) :
            for p in self.parameters() :
                p.data += -alpha*p.grad 
        ypred = [ n(x) for x in xs ]
        loss = sum((yi - yi_pred)**2 for yi, yi_pred in zip(ys, ypred))
        return loss , ypred



#initialisation       
n = MLP(3 , [3 , 2 , 1]) 

#data
xs = [[2 , 3 , 5],
      [4 , 2 , 0],
      [1 , 2 , 3]]

ys =  [-1 , 0 ,1]



