import numpy as np
import abc

class ActivationFunction:
    """
        Abstract class that can be inherited to defines different types of Activation Functions.
    """
    

    @abc.abstractmethod
    def tbd(self) -> ...:
        """
            
        """
        print('Error: cannot instantiate/use the default ActivationFunction class - use a base class that overrides tbd()!')
        return None
