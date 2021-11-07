
import numpy as np
from .ActivationFunction import ActivationFunction

class ReLU(ActivationFunction):
    """
        ReLU activation function
    """
    
    def name(self) -> str:
        return 'ReLU'

    def __call__(self, x: float) -> float:
        """
            Returns f(x)
        """
        return np.maximum(x, 0)

    def d(self, x: float) -> float:
        """
            Returns f'(x)
            Technically the derivative at x = 0 is undefined, but we return 1 here
        """
        return (x >= 0) * 1
