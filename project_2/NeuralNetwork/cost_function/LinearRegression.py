
from autograd.differential_operators import elementwise_grad
import numpy as np
from autograd import elementwise_grad as egrad
from .CostFunction import CostFunction

class LinearRegression(CostFunction):
    
    def __init__(self, X: np.matrix, y: np.matrix):
        """
            Initiates LinearRegerssion class
            Parameters
                X (np.matrix): design matrix
                y (np.matrix): target values
        """
        self.X = X
        self.y = y
        self.n = y.shape[0]
        self.n_features = X.shape[1]
        self.calculated_grad = False
         
    def C(self, beta: np.matrix, indx: np.matrix = np.matrix([])) -> np.matrix:
        """
            Returuns the value of the cost function at a new beta values
            Parameters:
                beta (np.matrix): features vector
        """
        if indx.size == 0:
            return np.power((self.X @ beta - self.y), 2) / self.n
        return np.power((self.X[indx] @ beta - self.y[indx]), 2) / self.y[indx].shape[0]

    def grad_C(self, beta: np.matrix, indx: np.matrix = np.matrix([])) -> np.matrix:
        """
            Returns the gradient of the function evaluated at a new beta values, 
            using the analytical expression.
            Parameters:
                beta (np.matrix): features vector
        """
        if indx.size == 0:
            return (2 / self.n) * self.X.T @ (self.X @ beta - self.y) 
        return (2 / self.y[indx].shape[0]) * self.X[indx].T @ (self.X[indx] @ beta - self.y[indx])
        
