
import numpy as np
import abc

class Splitter:
    """
        Abstract class that can be inherited to define different types of splitting methods to be used by a Solver instance
    """

    @abc.abstractmethod
    def split(self, X: np.matrix, y: np.matrix) -> tuple:
        """
            Splits a design matrix and a set of data points into 2 or more sets
            Parameters:
                X (np.matrix): Design matrix to split up
                y (np.matrix): Data points to split up
            Returns:
                (tuple): Two-element tuple containing the split design matrices and data sets as dictionaries, at least with label OG
        """
        print('Error: cannot instantiate/use the default Splitter class - use a base class that overrides split()!')
        return None
