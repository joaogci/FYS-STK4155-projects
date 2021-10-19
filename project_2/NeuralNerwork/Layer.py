
import numpy as np
from abc import ABC, abstractmethod
from .activation.ActivationFunction import ActivationFunction

class Layer(ABC):
    """
        Abstract class that can be inherited to define two different types of Layers (hidden and output).
    """
    
    NAME = ""

    def __init__(self, activation_function: ActivationFunction) -> None:
        """
            Initialises the layer with a custom activation function
            Parameters:
                activation_function (ActivationFunction): The activation function to use for the layer
        """
        self._activationFn = activation_function

    @abstractmethod
    def tbd(self) -> ...:
        """

        """
        print('\033[91mError: cannot instantiate/use the default Layer class - use a base class that overrides tbd()!\033[0m')
        return None
