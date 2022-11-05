import random
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
  
  def __init__(self, n_inputs: int) -> None:
    
    self.w = np.random.uniform(-1, 1, [n_inputs])
    self.b = np.random.uniform(-1, 1, 1)

  def __repr__(self) -> str:
    return f"<Neuron {self.w} {self.b}>"

  def __call__(self, x):
    assert x.shape == self.w.shape, f"Input should have shape {self.w.shape}"
    print(np.multiply(self.w, x))
    print(np.multiply(self.w, x) + self.b)
    z = np.sum(np.multiply(self.w, x))
    print(z)
    a = np.tanh(z)
    return a

n = Neuron(3)
out = n(np.array([1, 2, 3]))
print(out)

