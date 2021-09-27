
import numpy as np
from Scaler import Scaler

class StandardScaler(Scaler):
    """
        Applies standard scaling to the design matrix, by subtracting the mean and dividing by the standard deviation
    """
    
    def __init__(self, with_std: bool = True):
        self._with_std = with_std

    
    def prepare(self, X: np.matrix):
        """
            Samples the basis design matrix to obtain the mean and standard deviation
            Parameters:
                X (np.matrix): Principal design matrix
        """
        
        if X.shape[1] == 1:
            self._mean_value = np.mean(X[:, 0])
            self._standard_deviation = np.std(X[:, 0]) 
        else:
            self._mean_value = np.zeros(X.shape[1] - 1)
            self._standard_deviation = np.zeros(X.shape[1] - 1)
            
            for i in range(1, X.shape[1]):
                self._mean_value[i - 1] = np.mean(X[:, i])
                self._standard_deviation[i - 1] = np.std(X[:, i])

    def scale(self, X: np.matrix) -> np.matrix:
        """
            Scales the design matrix by subtracting the mean and dividing by the std deviation
            Parameters:
                X (np.matrix): Design matrix to scale
        """
        
        X_scaled = np.ones(X.shape)
        
        if X.shape[1] == 1:
            if self._with_std:
                X_scaled[:, 0] = (X[:, 0] - self._mean_value) / self._standard_deviation
            else:
                X_scaled[:, 0] = (X[:, 0] - self._mean_value) 
        else:
            if self._with_std:
                for i in range(1, X.shape[1]):
                    X_scaled[:, i] = (X[:, i] - self._mean_value[i - 1]) / self._standard_deviation[i - 1]
            else:
                for i in range(1, X.shape[1]):
                    X_scaled[:, i] = (X[:, i] - self._mean_value[i - 1])
            
        return X_scaled
