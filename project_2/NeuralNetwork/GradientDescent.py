
import numpy as np
import matplotlib.pyplot as plt
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class GradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        
    def optimize(self, eta: float, tol: float = 1e-7, iter_max: int = int(1e5)) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        theta = np.zeros(self.n_features)
        self.MSE = list()
        self.eta = eta
        
        for epoch in range(1, iter_max + 1):
            dif = - self.eta * self.cost_function.grad_C(theta)
            if np.linalg.norm(dif) <= tol:
                print()
                self.plot_MSE()
                break
            
            theta = theta + dif
            self.MSE.append(self.cost_function.MSE(theta))
            print(epoch, end='\r')
        
        self.plot_MSE()
        return theta
    
    def plot_MSE(self):
        """
            Plots MSE as a function of epochs
        """
        plt.figure("MSE vs epochs - GD") 
        
        plt.plot(range(1, len(self.MSE)+1), self.MSE, label=f"eta={self.eta}")
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        # plt.show()