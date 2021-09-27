
import numpy as np
import abc

class Resampling:
    """
        Abstract class that can be inherited to define different types of ressampling techniques to be fed to a Solver instance
    """
    

    @abc.abstractmethod
    def resample(self) -> ...:
        """
            Resamples data
            Called by the Solver the Resampler is attached to
            Should be overloaded in child classes
            Parameters:
        """
        print('Error: cannot instantiate/use the default Ressampling class - use a base class that overrides resample()!')
        return None
