import numpy as np 

class Adam:
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, params, grads):
        pid = id(params)
        if pid not in self.m:
            self.m[pid] = np.zeros_like(params)
            self.v[pid] = np.zeros_like(params)
            self.t[pid] = 0

        self.t[pid] += 1
        self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grads
        self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m[pid] / (1 - self.beta1 ** self.t[pid])
        v_hat = self.v[pid] / (1 - self.beta2 ** self.t[pid])

        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params




        
