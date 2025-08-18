import numpy as np 

class Adam:
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1              # decay rate for first moment (mean)
        self.beta2 = beta2              # decay rate for second moment (variance)
        self.epsilon = epsilon          # small constant to avoid division by zero
        
        self.first_moment = {}          
        self.second_moment = {}         
        self.time_step = {}            

    def update(self, parameters, gradients):
        param_id = id(parameters)

        if param_id not in self.first_moment:
            self.first_moment[param_id] = np.zeros_like(parameters)
            self.second_moment[param_id] = np.zeros_like(parameters)
            self.time_step[param_id]


        self.time_step[param_id] += 1
        self.first_moment[param_id] = (
            self.beta1 * self.first_moment[param_id] + (1 - self.beta1) * gradients
        )

        self.second_moment[param_id] = (
            self.beta2 * self.second_moment[param_id] + (1 - self.beta2) * (gradients ** 2)
        )

        # Bias correction for moments
        m_hat = self.first_moment[param_id] / (1 - self.beta1 ** self.time_step[param_id])
        v_hat = self.second_moment[param_id] / (1 - self.beta2 ** self.time_step[param_id])

        # Parameter update rule
        parameters -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters
 
