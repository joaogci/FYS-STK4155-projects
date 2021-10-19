
import numpy as np
from .ActivationFunction import ActivationFunction

class ReLU(ActivationFunction):
    """
        ReLU activation function
    """

    def __call__(self, x: float) -> float:
        """
            Returns f(x)
        """
        return 0 if x < 0 else x

    def d(self, x: float) -> float:
        """
            Returns f'(x)
            Technically the derivative at x = 0 is undefined, but we return 1 here
        """
        return 0 if x < 0 else 1
