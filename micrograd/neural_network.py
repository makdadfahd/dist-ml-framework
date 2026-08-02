from engine import Value
import random
#solved the problem, it was the way how i imported enginer (from engine import Value) instead of (import engine)

class Neuron :
    def __init__(self,n_input) :
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.b = Value(random.uniform(-1 , 1))

    def __call__(self,x) :
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        output = act.tanh()
        #didn't understand why i can't do act.engine.Value.tanh()
        return output

class Layer : 
    def __init__(self, n_inputs, n_outputs) :
        # the number of neurons is the same as the number of outputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x) :
        outputs = [neuron(x) for neuron in self.neurons ] 
        #started to master using for loops at the same line :)
        return outputs[0] if len(outputs) == 1 else outputs

class MLP : 
    def __init__(self,n_inputs, n_outputs) :
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)]
        #fixed a bug of i out of range : by decreasing the length of the size by 1 

    def __call__(self, x) :
        for layer in self.layer :
            x = layer(x)
        return x 
        
    

n = MLP(3 , [2 , 3 , 1])
xs = [
    [2 , 3 , 5],
    [4 , 2 , 0],
    [1 , 2 , 3]
]
#alwats forgetting the comma in multi-demensional arrays :(
ys = [ -1 , 0 , 1 ]

ypred = [ n(x) for x in xs ]

loss = sum((yi - yi_pred)**2 for yi, yi_pred in zip(ys, ypred))
#still can't know how to display the data instead if the whole value
#why can't i use loss.data or engine.Value.loss.data or something like that

loss.backward()

print(n.layers[0].neurons[0].w[2].grad)

