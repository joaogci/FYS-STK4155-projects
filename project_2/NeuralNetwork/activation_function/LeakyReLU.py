
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
        return np.multiply((x >= 0), x) + np.multiply((x < 0), x * self._alpha)

    def d(self, x: float) -> float:
        """
            Returns f'(x)
        """
        return (x >= 0) * 1 + (x < 0) * self._alpha
