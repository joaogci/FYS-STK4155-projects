
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
        Abstract class that can be inherited to define different Optimizers.
    """
    
    @abstractmethod
    def optimize(self) -> ...:
        """
            
        """
        print('Error: cannot instantiate/use the default Optimzer class - use a base class that overrides tbd()!')
        return None
    
    @abstractmethod
    def optimize_autograd(self) -> ...:
        """
            
        """
        print('Error: cannot instantiate/use the default Optimzer class - use a base class that overrides tbd()!')
        return None
