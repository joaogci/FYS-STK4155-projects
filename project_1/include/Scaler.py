
import numpy as np
import abc

class Scaler:
    """
        Component that can be added to a Solver instance to provide scaling of the input data
    """

    @abc.abstractmethod
    def prepare(self, X: np.matrix):
        """
            Should be overridden in child classes
            Prepares the scaler by looking at the initial/overall design matrix
            Parameters:
                X (np.matrix): The design/feature matrix to take as basis for scaling
        """
        print('Error: cannot instantiate/use the default Scaler class - use a base class that overrides prepare()!')
        return None

    @abc.abstractmethod
    def scale(self, X: np.matrix) -> np.matrix:
        """
            Should be overridden in child classes
            Scales the design matrix X
            Parameters:
                X (np.matrix): The design/feature matrix to apply scaling to
            Returns:
                (np.matrix): The scaled design matrix
        """
        print('Error: cannot instantiate/use the default Scaler class - use a base class that overrides scale()!')
        return None
