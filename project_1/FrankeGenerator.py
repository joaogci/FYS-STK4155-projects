
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
        
    def __init__(self, a: float, b: float, h: float, random: bool = False, noise: float = 0):
        self._noise = noise
        if not random:
            self._x = np.arange(a, b, h)
            self._y = np.arange(a, b, h)
        else:
            self._x = np.sort((b - a) * np.random.rand(int((b - a) / h)) + a)
            self._y = np.sort((b - a) * np.random.rand(int((b - a) / h)) + a)
            

    def generate(self) -> tuple:

        if hasattr(self, '_franke'): # Data has already been generated, just return cached version
            return np.ravel(self._X), np.ravel(self._Y), np.ravel(self._franke)

        self._X, self._Y = np.meshgrid(self._x, self._y)
        
        # Generate data set
        term1 = 0.75*np.exp(-(0.25*(9*self._X-2)**2) - 0.25*((9*self._Y-2)**2))
        term2 = 0.75*np.exp(-((9*self._X+1)**2)/49.0 - 0.1*(9*self._Y+1))
        term3 = 0.5*np.exp(-(9*self._X-7)**2/4.0 - 0.25*((9*self._Y-3)**2))
        term4 = -0.2*np.exp(-(9*self._X-4)**2 - (9*self._Y-7)**2)
        self._franke = term1 + term2 + term3 + term4

        # Add optional noise
        self._franke += self._noise * np.random.normal(0, 1, self._franke.shape)

        return np.ravel(self._X), np.ravel(self._Y), np.ravel(self._franke)

    
    def plot(self, show: bool = True):
        """
            Renders a 3D plot of the initial data
        """
        if not hasattr(self, '_franke'):
            self.generate()

        fig = plt.figure('Franke function input data', figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self._X, self._Y, self._franke, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        
        ax.set_zlim(np.min(self._franke) - 0.3, np.max(self._franke) + 0.3)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title('Franke function input data')
        plt.xlabel('x')
        plt.ylabel('y')

        if show:
            plt.show()

