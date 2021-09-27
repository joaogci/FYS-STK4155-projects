
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
        
        if X.shape[1] == 1:
            self._median = np.median(X[:, 0])
            self._inter_quartile_range = np.percentile(X[:, 0], 75) - np.percentile(X[:, 0], 25) 
        else:
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
        
        X_scaled = np.ones(X.shape)
        
        if X.shape[1] == 1:
            X_scaled[: ,0] = (X[:, 0] - self._median) / self._inter_quartile_range
        for i in range(1, X.shape[1]):
            X_scaled[:, i] = (X[:, i] - self._median[i - 1]) / self._inter_quartile_range[i - 1]
        
        return X_scaled
