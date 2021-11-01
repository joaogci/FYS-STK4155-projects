
import numpy as np
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class StochasticGradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction, size_minibatches: int, t0: int, t1: int, rng: np.random.Generator):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        self.n_batches = cost_function.n // size_minibatches
        self.size_minibatches = size_minibatches
        self.t0 = t0
        self.t1 = t1
        self.eta = lambda t: self.t0 / (t + self.t1)
        self.rng = rng
         
    
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
        broken = False
        
        for epoch in range(1, iter_max + 1):
            for i in range(self.n_batches):
                k = self.rng.integers(self.n_batches)
                dif = - self.eta(epoch * self.n_batches + i) * self.cost_function.grad_C(theta, indx=np.arange(k*self.size_minibatches, (k+1)*self.size_minibatches, 1))
                if np.linalg.norm(dif) <= tol:
                    broken = True
                    break
            
                theta = theta + dif
            if broken:
                break
            print(epoch, end='\r')
            
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
        broken = False

        for epoch in range(1, iter_max + 1):
            for i in range(self.n_batches):
                k = self.rng.integers(self.n_batches)
                dif = - self.eta(epoch * self.n_batches + i) * self.cost_function.grad_C_autograd(theta, indx=np.arange(k*self.size_minibatches, (k+1)*self.size_minibatches, 1))
                if np.linalg.norm(dif) <= tol:
                    broken = True
                    break
                
                theta = theta + dif
            if broken:
                break
            print(epoch, end='\r')
            
        return theta
    
