
import numpy as np
from abc import ABC, abstractmethod

class CostFunction(ABC):
    """
        Abstract class that can be inherited to define different Optimizers.
    """
    
    n_features = 0
    
    @abstractmethod
    def __init__(self, X: np.matrix, y: np.matrix):
        """
            Initiates the CostFunction class 
            Parameters
                X (np.matrix): design matrix
                y (np.matrix): target values
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides __init__()!')
        return None
    
    @abstractmethod
    def C(self, beta: np.matrix) -> np.matrix:
        """
            Calls the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides C()!')
        return None
    
    @abstractmethod
    def grad_C(self, beta: np.matrix) -> np.matrix:
        """
            Class the gradient of the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides grad_C()!')
        return None
    
    @abstractmethod
    def grad_C_autograd(self, beta: np.matrix) -> np.matrix:
        """
            Class the gradient of the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides grad_C_autograd()!')
        return None
