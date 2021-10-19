import numpy as np
import abc

class Optimizer:
    """
        Abstract class that can be inherited to define different Optimizers.
    """
    

    @abc.abstractmethod
    def tbd(self) -> ...:
        """
            
        """
        print('Error: cannot instantiate/use the default Optimzer class - use a base class that overrides tbd()!')
        return None
