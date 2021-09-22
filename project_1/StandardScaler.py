
import numpy as np
from Scaler import Scaler

class StandardScaler(Scaler):
    """
        Applies standard scaling to the features, by subtracting the mean and dividing by the standard deviation
    """
    
    def prepare(self, X: np.matrix):
        """
            Samples the basis design matrix to obtain the mean and standard deviation
            Parameters:
                X (np.matrix): Principal design matrix
        """
        self._means = np.zeros() #TODO

    def scale(self, X: np.matrix) -> np.matrix:
        """
            Scales the design matrix by subtracting the mean and dividing by the std deviation
            Parameters:
                X (np.matrix): Design matrix to scale
        """
        #TODO
        for i in range(1, self.X_train.shape[1]):
            mean_value = np.mean(self.X_train[:, i])
            standard_deviation = np.std(self.X_train[:, i])
            
            self.X_train[:, i] = (self.X_train[:, i] - mean_value) / standard_deviation
            self.X_test[:, i] = (self.X_test[:, i] - mean_value) / standard_deviation

