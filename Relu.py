import numpy as np 

class Relu:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backprop(self, d_L_d_out):
        grad = d_L_d_out.copy()
        grad[self.last_input <= 0] = 0
        return grad
