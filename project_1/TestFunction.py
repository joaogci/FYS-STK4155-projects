from Regression import Regression
import numpy as np

class TestFunction:
    """
        Test function class to test regression models
        
        Parameters: 
            a (float): beginning of the interval for x and y
            b (float): end of the interval for x and y
            h (float): step between successive points
    """
        
    def __init__(self, a, b, h):
        self.x = (b - a) * np.random.rand(int((b - a) / h)) + a
            
    def data_set(self):        
        self.y = np.exp(-self.x**2) + 1.5 * np.exp(-(self.x-2)**2)
   
    def add_noise(self):
        self.y = self.y + np.random.normal(0, 0.5, self.x.shape)
        
    def initialize_regression(self):
        return Regression(self.x, self.y)
    


