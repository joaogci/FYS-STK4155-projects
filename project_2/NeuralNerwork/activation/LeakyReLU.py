
import numpy as np
from .ActivationFunction import ActivationFunction

class LeakyReLU(ActivationFunction):
    """
        Leaky ReLU activation function
    """

    def __init__(self, alpha: float = 5e-3):
        """
            Initialises leaky ReLU with small alpha
            Parameters:
                alpha (float): Should be fairly small; typically in 1e-3..1e-2
        """
        self._alpha = alpha

    def __call__(self, x: float) -> float:
        """
            Returns f(x)
        """
        return self._alpha * x if x < 0 else x

    def d(self, x: float) -> float:
        """
            Returns f'(x)
        """
        return self._alpha if x < 0 else 1
