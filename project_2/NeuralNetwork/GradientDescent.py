
import numpy as np
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class GradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction):
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        
    def optimize(self, eta: float, tol: float = 1e-7, iter_max: int = 1e5) -> np.matrix:
        beta = np.zeros((self.n_features, 1))
        
        for i in range(iter_max):
            dif = - eta * self.cost_function.grad_C(beta)
            if np.abs(dif) <= tol:
                break
            
            beta = beta + dif
            
        return beta

    
    