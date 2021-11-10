
from math import e
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from .Optimizer import Optimizer
from ..cost_function.CostFunction import CostFunction

class StochasticGradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction, size_minibatches: int):
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
        self.eta = eta
        error_list = list()
       
        print()
        print("-- Stochastic Gradient Descent --")
        print()
        
        error_list.append(self.cost_function.error(theta)) 
       
        for epoch in range(1, iter_max + 1):
            self.cost_function.perm_data(self.rng)
            
            for i in range(self.n_batches):
                k = self.rng.integers(self.n_batches)
                grad = self.cost_function.grad_C(theta, indx=np.arange(k*self.size_minibatches, (k+1)*self.size_minibatches, 1))
                theta = theta - self.eta * grad
                
            error = self.cost_function.error(theta)
            error_list.append(error)
            print(f"[ Epoch: {epoch}/{iter_max}; {self.cost_function.error_name()}: {error} ]", end='\r')
            
            if epoch >= 5 and np.abs(np.mean(error_list[-5:] - error)) <= tol:
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
        
    def optimize_learning_schedule(self, eta: Callable, random_state: int, tol: float = 1e-7, iter_max: int = int(1e5), verbose: bool = False) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                eta (function): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        self.rng = np.random.default_rng(np.random.MT19937(seed=random_state))
        theta = self.rng.random(self.n_features)
        self.eta = eta
        error_list = list()
       
        print()
        print("-- Stochastic Gradient Descent --")
        print()
        
        error_list.append(self.cost_function.error(theta)) 
       
        for epoch in range(1, iter_max + 1):
            self.cost_function.perm_data(self.rng)
            
            for i in range(self.n_batches):
                # k = self.rng.integers(self.n_batches)
                grad = self.cost_function.grad_C(theta, indx=np.arange(i*self.size_minibatches, (i+1)*self.size_minibatches, 1))
                theta = theta - self.eta(epoch - 1) * grad
                
            error = self.cost_function.error(theta)
            error_list.append(error)
            print(f"[ Epoch: {epoch}/{iter_max}; {self.cost_function.error_name()}: {error} ]", end='\r')
            
            if epoch >= 5 and np.abs(np.mean(error_list[-5:] - error)) <= tol:
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
        plt.figure("MSE vs epochs - SGD") 
        
        plt.plot(range(1, len(self.MSE)+1), self.MSE, label=f"eta={self.eta}")
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        # plt.show()
        
        
        
