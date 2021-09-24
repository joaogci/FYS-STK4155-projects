
import numpy as np
from DataGenerator import DataGenerator

class ExponentialGenerator(DataGenerator):
    """
        A Data Generator that can be used to generate randomized data that follows 1D exponential function
    """

    def __init__(self, count: int = 100, min_x: float = 0, max_x: float = 1, degree: int = 2, noise: float = 1, normalise: bool = True):
        self._count = count
        self._min_x = min_x
        self._max_x = max_x
        self._degree = degree
        self._noise = noise
        self._normalise = normalise


    def generate(self, rng: np.random.Generator) -> tuple:
        """
            Generates a set of noisy data from a polynomial
        """

        # Generate x points between min_x and max_x
        x = rng.uniform(self._min_x,self._max_x,(self._count,1))
        y = np.zeros((self._count, 1))

        # Generate polynomial coefficients and compute data points
        factors = rng.random((self._degree + 1, 1)) * 20 - 10
        for i in range(0, self._degree + 1):
            y += factors[i] * np.exp(factors[i] * x ** i)
        
        # Optionally normalize y between 0..1 or -1..1
        # (otherwise MSE becomes ridiculous if the range is already much bigger than 0..1)
        if self._normalise:
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Add noise
        y += rng.normal(0, 1, (self._count, 1)) * self._noise

        # Return data (2D data)
        return x, y
