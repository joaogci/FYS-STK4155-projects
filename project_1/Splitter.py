
import numpy as np
import abc

class Splitter:
    """
        Abstract class that can be inherited to define different types of splitting methods to be used by a Solver instance
    """

    @abc.abstractmethod
    def split(self, X: np.matrix, y: np.matrix) -> list:
        """
            Splits a design matrix and a set of data points into 2 or more sets
            Parameters:
                X (np.matrix): Design matrix to split up
                y (np.matrix): Data points to split up
            Returns:
                (list): The split design matrices and data sets as a dictionary of InputSets, at least with label full
        """
        print('Error: cannot instantiate/use the default Splitter class - use a base class that overrides split()!')
        return None
    
    @abc.abstractmethod
    def prediction_sources(self) -> list:
        """
            Should be overridden in child classes
            Gives the list of keys to use as prediction sets; each prediction set will be used to generate its own beta value in the Solver
            Returns:
                (PredictionSource[]): List of source sets to use into the split dicts to obtain the prediction sets (typically the training set(s))
        """
        print('Error: cannot instantiate/use the default Splitter class - use a base class that overrides prediction_sources()!')
        return None
