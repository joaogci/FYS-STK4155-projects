
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
        Abstract class that can be inherited to define different Optimizers.
    """
    
    @abstractmethod
    def optimize(self, eta: float, random_state: int, tol: float = 1e-7, iter_max: int = int(1e5)) -> ...:
        """
            Finds the minimum of the inpute CostFunction using the analytical expression for the gradient.
            If there is no analytical expression for the gradient, it uses autograd. 
        """
        print('Error: cannot instantiate/use the default Optimzer class - use a base class that overrides optimize()!')
        return None
   
    @abstractmethod
    def plot_MSE(self):
        """
            Plots MSE over the number of epochs.
        """ 
        print('Error: cannot instantiate/use the default Optimzer class - use a base class that overrides plot_MSE()!')
        return None
