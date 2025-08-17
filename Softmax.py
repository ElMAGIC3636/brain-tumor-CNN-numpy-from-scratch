import numpy as np

class Softmax:
    def __init__(self, input_len, nodes, l2_lambda=1e-4):
        scale = np.sqrt(2.0 / input_len)
        self.weights = np.random.randn(input_len, nodes) * scale
        self.biases = np.zeros(nodes)
        self.l2_lambda = l2_lambda

    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals - np.max(totals, axis=1, keepdims=True))
        output = exp / np.sum(exp, axis=1, keepdims=True)
        self.last_output = output
        return output

    def backprop(self, d_L_d_out, optimizer_w, optimizer_b):
        d_t = d_L_d_out

        d_L_d_w = np.dot(self.last_input.T, d_t) / d_t.shape[0]
        d_L_d_b = np.sum(d_t, axis=0) / d_t.shape[0]
        d_L_d_input = np.dot(d_t, self.weights.T)

        d_L_d_w += self.l2_lambda * self.weights

        self.weights = optimizer_w.update(self.weights, d_L_d_w)
        self.biases = optimizer_b.update(self.biases, d_L_d_b)
        return d_L_d_input
