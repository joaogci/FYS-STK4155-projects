
import numpy as np
from .ActivationFunction import ActivationFunction

class Linear(ActivationFunction):
    """
        Linear (passthrough) activation function
        To be used as activation function for the output layer in regression problems
    """

    def __call__(self, x: float) -> float:
        """
            Returns x (passthrough)
        """
        return x

    def d(self, x: float) -> float:
        """
            Returns the derivative of y = x
        """
        return 1 # (careful, this is a really expensive function to run computationally! use at your own risk - might need to parallelise)
