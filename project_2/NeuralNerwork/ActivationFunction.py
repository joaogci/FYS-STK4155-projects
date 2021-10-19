
import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """
        Abstract class that can be inherited to defines different types of Activation Functions.
    """
    

    @abstractmethod
    def __call__(self, x: float) -> float:
        """
            Returns the result of the activation function at x
            Parameters:
                x (float): The x-coordinate at which to evaluate the function
            Returns:
                (float): The value f(x)
        """
        print('\033[91mError: cannot instantiate/use the default ActivationFunction class - use a base class that overrides __call__()!\033[0m')
        return None
