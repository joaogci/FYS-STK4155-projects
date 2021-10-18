import numpy as np
import abc

class Layer:
    """
        Abstract class that can be inherited to define two different types of Layers (hidden and output).
    """
    
    NAME = ""    

    @abc.abstractmethod
    def tbd(self) -> ...:
        """

        """
        print('Error: cannot instantiate/use the default Layer class - use a base class that overrides tbd()!')
        return None
