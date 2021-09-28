
import numpy as np
from Scaler import Scaler

class MinMaxScaler(Scaler):
    """
        Applies min-max scaling to the design matrix by remapping from min..max to 0..1
    """

    def _subtractor(self, x: np.matrix) -> np.matrix:
        return np.min(x)

    def _divisor(self, x: np.matrix) -> np.matrix:
        return np.max(x) - np.min(x)
