import numpy as np

def sigmoid():
    func = lambda x: 1 / (1 + np.exp(-x))
    d_func = lambda x: np.multiply(func(x), (1 - func(x)))
    return func, d_func

def ReLU():
    func = lambda x: np.maximum(x, 0)
    d_func = lambda x: (x >= 0) * 1
    return func, d_func

def leakyReLU(alpha: float):
    func = lambda x: np.multiply((x >= 0), x) + np.multiply((x < 0), x * alpha)
    d_func = lambda x: (x >= 0) * 1 + (x < 0) * alpha
    return func, d_func

def linear():
    func = lambda x: x
    d_func = lambda x: 1
    return func, d_func

def tanh():
    func = lambda x: np.tanh(x)
    d_func = lambda x: 1.0 - np.multiply(np.tanh(x), np.tanh(x))
    return func, d_func

def eLU(alpha):
    func = lambda x: np.multiply((x >= 0), x) + np.multiply((x < 0), (np.exp(x) - 1.0) * alpha)
    d_func = lambda x: (x >= 0) * 1 + np.multiply((x < 0), alpha * np.exp(x))
    return func, d_func

def soft_max():
    func = lambda x: np.exp(x - 7 - np.log(np.sum(np.exp(x)/1096.6331584284585992637202382881214324422191348336131437827392407, axis=0, keepdims=True)))
    d_func = lambda x: np.multiply((1.0 - func(x)), func(x))
    return func, d_func
