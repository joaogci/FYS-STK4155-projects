
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.sparse.construct import rand
from .Optimizer import Optimizer
from ..cost_function.CostFunction import CostFunction

class NewtonMethod(Optimizer):
    
    def __init__(self, cost_function: CostFunction):
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                cost_function (CostFunction): cost function to minimize
        """
        self.cost_function = cost_function
        self.n_features = cost_function.n_features
        
    def optimize(self, eta: float, random_state: int, tol: float = 1e-7, iter_max: int = int(1e5), verbose: bool = False) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        self.rng = np.random.default_rng(np.random.MT19937(seed=random_state))
        theta = self.rng.random(self.n_features)
        error_list = list()
        
        print()
        print("-- Newton's Method --")
        print()        
        
        error_list.append(self.cost_function.error(theta))
        
        for epoch in range(1, iter_max + 1):
            grad = self.cost_function.grad_C(theta)
            theta = theta - np.linalg.pinv(self.cost_function.hess_C(theta)) @ grad
            
            error = self.cost_function.error(theta)
            error_list.append(error)
            print(f"[ Epoch: {epoch}/{iter_max}; {self.cost_function.error_name()}: {error} ]", end='\r')
            
            if epoch >= 5 and np.abs(np.mean(error_list[-5:]) - error) <= tol:
                print()
                print(f"[ Finished training with error: {self.cost_function.error(theta)} ]")
                if verbose:
                    return theta, epoch, np.array(error_list)
                return theta
                
        print()
        print(f"[ Finished training with error: {self.cost_function.error(theta)} ]")
        if verbose:
            return theta, epoch, np.array(error_list)
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