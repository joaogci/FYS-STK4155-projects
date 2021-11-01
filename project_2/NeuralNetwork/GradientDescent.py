
import numpy as np
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class GradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction, eta: float):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        self.eta = eta
        
    def optimize(self, tol: float = 1e-7, iter_max: int = int(1e5)) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        theta = np.zeros(self.n_features)
        
        for epoch in range(1, iter_max + 1):
            dif = - self.eta * self.cost_function.grad_C(theta)
            if np.linalg.norm(dif) <= tol:
                break
            
            theta = theta + dif
            
        return theta
    
    def optimize_autograd(self, tol: float = 1e-7, iter_max: int = int(1e5)) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        theta = np.zeros(self.n_features)        
        
        for epoch in range(1, iter_max + 1):
            dif = - self.eta * self.cost_function.grad_C_autograd(theta)
            if np.linalg.norm(dif) <= tol:
                break
            
            theta = theta + dif
            
        return theta
