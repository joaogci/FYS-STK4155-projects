
import numpy as np
import abc

class Scaler:
    """
        Component that can be added to a Solver instance to provide scaling of the input data
        Performs scaling of the form ([input] - [subtractor]) / [divisor], where [subtractor] and [divisor] depend on the child classes
    """


    @abc.abstractmethod
    def _subtractor(self, x: np.matrix) -> np.matrix:
        """
            Should be overridden in child classes
            Given the matrix to scale, returns the value that should be subtracted from each row
            Parameters:
                x (np.matrix): The matrix to take as basis for scaling
        """
        print('Error: cannot instantiate/use the default Scaler class - use a base class that overrides _subtractor()!')
        return None

    @abc.abstractmethod
    def _divisor(self, x: np.matrix) -> np.matrix:
        """
            Should be overridden in child classes
            Given the matrix to scale, returns the value that should be divided from each row
            Parameters:
                x (np.matrix): The matrix to take as basis for scaling
        """
        print('Error: cannot instantiate/use the default Scaler class - use a base class that overrides _divisor()!')
        return None


    def prepare(self, x: np.matrix):
        """
            Prepares the scaler by looking at the initial/overall input matrix, sampling it to obtain the subtractor and divisor
            Parameters:
                x (np.matrix): The matrix to take as basis for scaling
        """
        if x.shape[1] == 1:
            self._subtractor_val = self._subtractor(x[:, 0])
            self._divisor_val = self._divisor(x[:, 0]) 
        else:
            self._subtractor_val = np.zeros(x.shape[1] - 1)
            self._divisor_val = np.zeros(x.shape[1] - 1)
            
            for i in range(1, x.shape[1]):
                self._subtractor_val[i - 1] = self._subtractor(x[:, i])
                self._divisor_val[i - 1] = self._divisor(x[:, i])

    def scale(self, x: np.matrix) -> np.matrix:
        """
            Scales the matrix x
            Parameters:
                x (np.matrix): The matrix to apply scaling to
            Returns:
                (np.matrix): The scaled matrix
        """
        x_scaled = np.ones(x.shape)
        
        if x.shape[1] == 1:
            x_scaled[:, 0] = (x[:, 0] - self._subtractor_val) / self._divisor_val
        else:
            for i in range(1, x.shape[1]):
                x_scaled[:, i] = (x[:, i] - self._subtractor_val[i - 1]) / self._divisor_val[i - 1]

        return x_scaled
