
import numpy as np
from Scaler import Scaler

class MinMaxScaler(Scaler):
    """
        Applies min-max scaling to the design matrix by remapping from min..max to 0..1
    """
    
    def prepare(self, X: np.matrix):
        """
            Samples the basis design matrix to obtain the min and max
            Parameters:
                X (np.matrix): Principal design matrix
        """
        
        self._min = np.zeros(X.shape[1] - 1)
        self._max = np.zeros(X.shape[1] - 1)
        
        for i in range(1, X.shape[1]):
            self._min[i - 1] = np.min(X[:, i])
            self._max[i - 1] = np.max(X[:, i])

    def scale(self, X: np.matrix) -> np.matrix:
        """
            Scales the design matrix by remapping from min..max to 0..1
            Parameters:
                X (np.matrix): Design matrix to scale
        """
        
        for i in range(1, X.shape[1]):
            X[:, i] = (X[:, i] - self._min) / (self._max - self._min)
        
        return X
