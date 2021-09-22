
import numpy as np
from Scaler import Scaler

class RobustScaler(Scaler):
    """
        Applies robust scaling to the design matrix by subtracting the median and dividing by the inter quartile range
    """
    
    def prepare(self, X: np.matrix):
        """
            Samples the basis design matrix to obtain the median and quartile range
            Parameters:
                X (np.matrix): Principal design matrix
        """
        
        self._median = np.zeros(X.shape[1] - 1)
        self._inter_quartile_range = np.zeros(X.shape[1] - 1)
        
        for i in range(1, X.shape[1]):
            self._median[i - 1] = np.median(X[:, i])
            self._inter_quartile_range[i - 1] = np.percentile(X[:, i], 75) - np.percentile(X[:, i], 25)

    def scale(self, X: np.matrix) -> np.matrix:
        """
            Scales the design matrix by subtracting the median and dividing by the inter quartile range
            Parameters:
                X (np.matrix): Design matrix to scale
        """
        
        for i in range(1, X.shape[1]):
            X[:, i] = (X[:, i] - self._median) / self._inter_quartile_range
        
        return X
