import numpy as np 

# Relu Activation
def cross_entropy(pred, target):
    m = target.shape[0]
    log_pred = -np.log(np.clip(pred, 1e-12, 1.0))
    loss = np.sum(target * log_pred) / m
    return loss

def cross_entropy_grad(pred, target):
    m = target.shape[0]
    return (pred - target) / m
