
import numpy as np
from Scaler import Scaler

class StandardScaler(Scaler):
    """
        Applies standard scaling to the design matrix, by subtracting the mean and dividing by the standard deviation
    """
    
    def __init__(self, with_std: bool = True):
        self._with_std = with_std
    
    def _subtractor(self, x: np.matrix) -> np.matrix:
        return np.mean(x)
    
    def _divisor(self, x: np.matrix) -> np.matrix:
        # Use 1 as the divisor if with_std==False, which is the same as just not dividing by anything
        return np.std(x) if self._with_std else 1
