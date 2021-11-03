
import numpy as np
from abc import ABC, abstractmethod

class CostFunction(ABC):
    """
        Abstract class that can be inherited to define different Optimizers.
    """
    
    n_features = 0
    n = 0
    
    @abstractmethod
    def __init__(self, X_train: np.matrix, y_train: np.matrix, X_test: np.matrix, y_test: np.matrix):
        """
            Initiates the CostFunction class 
            Parameters
                X_train (np.matrix): design train matrix
                y_train (np.matrix): target train values
                X_test (np.matrix): design test matrix
                y_test (np.matrix): target test values
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides __init__()!')
        return None
    
    @abstractmethod
    def C(self, beta: np.matrix, indx: np.array = None) -> np.matrix:
        """
            Calls the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides C()!')
        return None
    
    @abstractmethod
    def grad_C(self, beta: np.matrix, indx: np.array = None) -> np.matrix:
        """
            Class the gradient of the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides grad_C()!')
        return None

    @abstractmethod
    def grad_C_nn(self, y_data: np.matrix, y_tilde: np.matrix) -> np.matrix:
        """
            Class the gradient of the cost function
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides grad_C_nn()!')
        return None
    
    @abstractmethod
    def MSE(self, beta: np.matrix) -> np.matrix: 
        """
            Computes the Mean Squared Error
        """
        print('Error: cannot instantiate/use the default CostFunction class - use a base class that overrides MSE()!')
        return None 
    