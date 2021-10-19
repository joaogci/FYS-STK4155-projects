
import numpy as np
from .ActivationFunction import ActivationFunction

class ELU(ActivationFunction):
    """
        eLU activation function
    """

    def __init__(self, alpha: float = 5e-3):
        """
            Initialises eLU with small alpha
            Parameters:
                alpha (float): Should be fairly small
        """
        self._alpha = alpha

    def __call__(self, x: float) -> float:
        """
            Returns f(x)
        """
        return self._alpha * (np.exp(x) - 1.0) if x < 0 else x

    def d(self, x: float) -> float:
        """
            Returns f'(x)
        """
        return self._alpha * np.exp(x) if x < 0 else 1
