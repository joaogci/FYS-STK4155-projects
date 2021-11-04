
import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """
        Abstract class that can be inherited to defines different types of Activation Functions.
    """
    

    @abstractmethod
    def __call__(self, x: float) -> float:
        """
            Returns the evaluation of the activation function at x values
            Parameters:
                x (float|np.matrix): The x-coordinate(s) at which to evaluate the function
            Returns:
                (float|np.matrix): The value(s) f(x)
        """
        print('\033[91mError: cannot instantiate/use the default ActivationFunction class - use a base class that overrides __call__()!\033[0m')
        return None
    
    @abstractmethod
    def d(self, x: float) -> float:
        """
            Returns the evaluation of the first derivative of the activation function at x
            Parameters:
                x (float|np.matrix): The x-coordinate(s) at which to evaluate the first derivative
            Returns:
                (float|np.matrix): The value(s) f'(x)
        """
        print('\033[91mError: cannot instantiate/use the default ActivationFunction class - use a base class that overrides d()!\033[0m')
        return None
