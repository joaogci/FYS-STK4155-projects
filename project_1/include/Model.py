
import numpy as np
import abc

class Model:
    """
        Abstract class that can be inherited to define different types of models to interpolate data points, used as a Solver component
    """

    """
        Override per implementation to display the name of the model
    """
    name = '<none>'

    @abc.abstractmethod
    def interpolate(self, design_matrix: np.matrix, y: np.matrix) -> np.matrix:
        """
            Given a design matrix and a (training) data set, returns the beta feature matrix
        """
        print('Error: cannot instantiate/use the default Model class - use a base class that overrides interpolate()!')
        return None
