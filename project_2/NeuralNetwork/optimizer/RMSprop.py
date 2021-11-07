
import numpy as np
import matplotlib.pyplot as plt
from .Optimizer import Optimizer
from ..cost_function.CostFunction import CostFunction

class RMSprop(Optimizer):
    
    def __init__(self, cost_function: CostFunction):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
            
    def optimize(self, eta: float, random_state: int, tol: float = 1e-7, iter_max: int = int(1e5)) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        self.rng = np.random.default_rng(np.random.MT19937(seed=random_state))
        theta = self.rng.random(self.n_features)
        self.eta = eta
        prev_error = 1000000
        
        beta = 0.9
        epsilon = 1e-8
        s = 0
        
        print()
        print("-- RMSprop --")
        print()
        
        for epoch in range(1, iter_max + 1):
            g = self.cost_function.grad_C(theta)
            s = beta * s + (1 - beta) * g**2 
            
            grad = g / (np.sqrt(s + epsilon))
            theta = theta - eta *  grad
                        
            error = self.cost_function.error(theta)
            print(f"[ Epoch: {epoch}/{iter_max}; {self.cost_function.error_name()}: {error} ]")
            
            if np.abs(prev_error - error) <= tol:
                print()
                print(f"[ Finished training with error: {self.cost_function.error(theta)} ]")
                return theta
            prev_error = error
        
        print()
        print(f"[ Finished training with error: {self.cost_function.error(theta)} ]")
        return theta
    
    def plot_MSE(self):
        """
            Plots MSE as a function of epochs
        """
        plt.figure("MSE vs epochs - RMSprop") 
        
        plt.plot(range(1, len(self.MSE)+1), self.MSE, label=f"eta={self.eta}")
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        # plt.show()
        
        
        
