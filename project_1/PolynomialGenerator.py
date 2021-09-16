
import numpy as np
from DataGenerator import DataGenerator

class PolynomialGenerator(DataGenerator):
    """
        A Data Generator that can be used to generate randomized data that follows an nth order 1D polynomial
    """

    def __init__(self, count: int = 100, min_x: float = 0, max_x: float = 1, degree: int = 2, noise: float = 1):
        self._count = count
        self._min_x = min_x
        self._max_x = max_x
        self._degree = degree
        self._noise = noise


    def generate(self) -> tuple:
        """
            Generates a set of noisy data from a polynomial
        """

        # Generate x points between min_x and max_x
        x = np.random.rand(self._count, 1) * (self._max_x - self._min_x) + self._min_x
        x = np.sort(x, 0)
        y = np.zeros((self._count, 1))

        # Generate polynomial coefficients and compute data points
        factors = np.random.rand(self._degree + 1, 1) * 10 - 5
        for i in range(0, self._degree + 1):
            y += factors[i] * x ** i
        
        # @todo: optionally normalize y between 0..1 or -1..1 (otherwise MSE becomes ridiculous if the range is already -800..800 or something)

        # Add noise
        y += np.random.randn(self._count, 1) * self._noise

        # Return data (2D data)
        return x, y
