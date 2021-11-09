import numpy as np

def mse(target, pred):
    return np.mean((pred - target)**2)

def grad_mse(target, pred):
    return 2 * (pred - target) / target.shape[1]
    
def cross_entropy(target, pred):
    return - np.mean(- target * np.log(pred) - (1 - target) * np.log(1 - pred)) 

def grad_cross_entropy(target, pred):
    return - (target - pred) / target.shape[1]
