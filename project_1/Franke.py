import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Franke:
    """
        Franke function class
        
        Parameters: 
            a (double): beginning of the interval for x and y
            b (double): end of the interval for x and y
            h (double): step between successive points
            random (boolean): if True, use random inputs with the number of points specified by (b - a) / h 
        
        Methods: 
            function(): returns a numpy matrix with the values for the Franke function over a meshgrid of x and y
            plot(): 3D plot of the computed Franke function
    """
    
    def __init__(self):
        pass
    
    def __init__(self, a, b, h, random=False):
        if not random:
            self.x = np.arange(a, b, h)
            self.y = np.arange(a, b, h)
        else:
            self.x = np.sort((b - a) * np.random.rand(int((b - a) / h)) + a)
            self.y = np.sort((b - a) * np.random.rand(int((b - a) / h)) + a)
            
    def function(self):
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        term1 = 0.75*np.exp(-(0.25*(9*self.X-2)**2) - 0.25*((9*self.Y-2)**2))
        term2 = 0.75*np.exp(-((9*self.X+1)**2)/49.0 - 0.1*(9*self.Y+1))
        term3 = 0.5*np.exp(-(9*self.X-7)**2/4.0 - 0.25*((9*self.Y-3)**2))
        term4 = -0.2*np.exp(-(9*self.X-4)**2 - (9*self.Y-7)**2)
        
        self.franke = term1 + term2 + term3 + term4
        
        return self.franke
    
    def plot(self):
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self.X, self.Y, self.franke, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        
        ax.set_zlim(np.min(self.franke) - 0.3, np.max(self.franke) + 0.4)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

