
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from DataGenerator import DataGenerator

class FrankeGenerator(DataGenerator):
    """
        Franke function class
        
        Parameters: 
            a (float): beginning of the interval for x and y
            b (float): end of the interval for x and y
            h (float): step between successive points
            random (boolean): if True, use random inputs with the number of points specified by (b - a) / h 
            noise (float): Scale of noise to add to the results
    """
        
    def __init__(self, a: float, b: float, h: int, random: bool = False, noise: float = 0):
        self._a = a
        self._b = b
        self._h = h
        self._random = random
        self._noise = noise
            

    def generate(self, rng: np.random.Generator) -> tuple:

        if hasattr(self, '_franke'): # Data has already been generated, just return cached version
            return np.ravel(self._X), np.ravel(self._Y), np.ravel(self._franke)
            
        if not self._random:
            self._x = np.linspace(self._a, self._b, self._h)
            self._y = np.linspace(self._a, self._b, self._h)
        else:
            self._x = np.sort(rng.uniform(self._a,self._b,self._h))
            self._y = np.sort(rng.uniform(self._a,self._b,self._h))

        self._X, self._Y = np.meshgrid(self._x, self._y)
        
        # Generate data set
        term1 = 0.75*np.exp(-(0.25*(9*self._X-2)**2) - 0.25*((9*self._Y-2)**2))
        term2 = 0.75*np.exp(-((9*self._X+1)**2)/49.0 - 0.1*(9*self._Y+1))
        term3 = 0.5*np.exp(-(9*self._X-7)**2/4.0 - 0.25*((9*self._Y-3)**2))
        term4 = -0.2*np.exp(-(9*self._X-4)**2 - (9*self._Y-7)**2)
        self._franke = term1 + term2 + term3 + term4

        # Add optional noise
        self._franke += self._noise * rng.normal(0, 1, self._franke.shape)

        return np.ravel(self._X), np.ravel(self._Y), np.ravel(self._franke)

