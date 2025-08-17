import numpy as np 

class Flatten:
    def forward(self, input):
        self.last_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backprop(self, d_L_d_out):
        return d_L_d_out.reshape(self.last_shape)
