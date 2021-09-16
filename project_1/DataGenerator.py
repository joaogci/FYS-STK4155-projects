
import numpy as np
import abc

class DataGenerator:
    """
        Abstract class that can be inherited to define different types of data generation method to be fed to a Solver instance
    """
    

    @abc.abstractmethod
    def generate(self) -> tuple:
        """
            Generates and returns 2D or 3D data points
            Called by the Solver the DataGenerator is attached to
            Should be overloaded in child classes
        """
        print('Error: cannot instantiate/use the default DataGenerator class - use a base class that overrides generate()!')
        return None
