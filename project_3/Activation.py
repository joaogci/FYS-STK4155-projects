import autograd.numpy as np

def sigmoid():
    return lambda x: 1 / (1 + np.exp(-x))

def ReLU():
    return lambda x: np.maximum(x, 0)

def leakyReLU(alpha: float):
    func = lambda x: np.multiply((x >= 0), x) + np.multiply((x < 0), x * alpha)
    return func

def linear():
    return lambda x: x

def tanh():
    return lambda x: np.tanh(x)

def eLU(alpha):
    return lambda x: np.multiply((x >= 0), x) + np.multiply((x < 0), (np.exp(x) - 1.0) * alpha)

def soft_max():
    return lambda x: np.exp(x - 7 - np.log(np.sum(np.exp(x)/1096.6331584284585992637202382881214324422191348336131437827392407, axis=0, keepdims=True)))
