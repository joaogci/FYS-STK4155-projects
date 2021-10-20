
import numpy as np
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class StochasticGradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        
    def optimize(self, eta: float, tol: float = 1e-7, iter_max: int = 1e5) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        beta = np.zeros((self.n_features, 1))
        
        for i in range(iter_max):
            dif = - eta * self.cost_function.grad_C(beta)
            if np.abs(dif) <= tol:
                break
            
            beta = beta + dif
            
        return beta
    
    def optimize_autograd(self, eta: float, tol: float = 1e-7, iter_max: int = 1e5) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        beta = np.zeros((self.n_features, 1))
        
        for i in range(iter_max):
            dif = - eta * self.cost_function.grad_C_autograd(beta)
            if np.abs(dif) <= tol:
                break
            
            beta = beta + dif
            
        return beta
