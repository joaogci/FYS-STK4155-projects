
import numpy as np
from Scaler import Scaler

class RobustScaler(Scaler):
    """
        Applies robust scaling to the design matrix by subtracting the median and dividing by the inter quartile range
    """

    def _subtractor(self, x: np.matrix) -> np.matrix:
        return np.median(x)
    
    def _divisor(self, x: np.matrix) -> np.matrix:
        return np.percentile(x, 75) - np.percentile(x, 25)
