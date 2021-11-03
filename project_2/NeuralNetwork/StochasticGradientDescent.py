
import numpy as np
import matplotlib.pyplot as plt
from .Optimizer import Optimizer
from .cost_function.CostFunction import CostFunction

class StochasticGradientDescent(Optimizer):
    
    def __init__(self, cost_function: CostFunction, size_minibatches: int, rng: np.random.Generator):
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
        self.rng = rng
    
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
            for i in range(self.n_batches):
                k = self.rng.integers(self.n_batches)
                dif = - self.eta * self.cost_function.grad_C(theta, indx=np.arange(k*self.size_minibatches, (k+1)*self.size_minibatches, 1))

                if np.linalg.norm(dif, ord=np.inf) <= tol:
                    print()
                    self.plot_MSE()
                    return theta

                theta = theta + dif

            self.MSE.append(self.cost_function.MSE(theta))
            print(epoch, end='\r')
        
        self.plot_MSE()
        
        print("Need to increase iter_max to achieve desired tol.")
        return theta
    
    def optimize_eta_function(self, t0: float, t1: float, tol: float = 1e-7, iter_max: int = int(1e5)) -> np.matrix:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            Parameters:
                eta (float): learning rate
                tol (float): tolerance
                iter_max (int): maximum number of iterations
        """
        theta = np.zeros(self.n_features)
        self.MSE = list()
        self.eta = "function"
        self.eta_f = lambda t: t0 / (t + t1)
        
        for epoch in range(1, iter_max + 1):
            for i in range(self.n_batches):
                k = self.rng.integers(self.n_batches)
                dif = - self.eta_f(epoch * self.n_batches + i) * self.cost_function.grad_C(theta, indx=np.arange(k*self.size_minibatches, (k+1)*self.size_minibatches, 1))

                if np.linalg.norm(dif, ord=np.inf) <= tol:
                    print()
                    self.plot_MSE()
                    return theta

                theta = theta + dif

            self.MSE.append(self.cost_function.MSE(theta))
            print(epoch, end='\r')
        
        self.plot_MSE()
        
        print("Need to increase iter_max to achieve desired tol.")
        return theta
    
    def plot_MSE(self):
        """
            Plots MSE as a function of epochs
        """
        plt.figure("MSE vs epochs") 
        
        plt.plot(range(1, len(self.MSE)+1), self.MSE, label=f"eta={self.eta}")
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.legend()
        # plt.show()
        
        
        
