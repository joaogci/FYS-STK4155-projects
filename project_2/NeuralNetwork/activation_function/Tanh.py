
import numpy as np
from .ActivationFunction import ActivationFunction

class Tanh(ActivationFunction):
    """
        tanh activation function
    """
    
    def name(self) -> str:
        return 'Tanh'

    def __call__(self, x: float) -> float:
        """
            Returns f(x)
        """
        return np.tanh(x)

    def d(self, x: float) -> float:
        """
            Returns f'(x)
            1 - tanh(x)**2
        """
        t = np.tanh(x)
        return 1.0 - np.multiply(t, t)
