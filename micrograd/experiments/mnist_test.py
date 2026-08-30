import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import MLP 
from multi_dim_engine import Tensor

mnsit = MLP(784,[128,64,10])

mnsit.learn(64 , 10)

mnsit.pred(10000)