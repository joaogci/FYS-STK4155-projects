
import numpy as np
from .ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):
    """
        Sigmoid activation function
    """

    def __call__(self, x: float) -> float:
        """
            Returns the result of the sigmoid f(x)
        """
        return 1.0 / (1.0 + np.exp(-x))

    def d(self, x: float) -> float:
        """
            Returns the derivative of the sigmoid f'(x)
        """
        return self(x) * (1.0 - self(x))
