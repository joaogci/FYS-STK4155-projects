
import numpy as np
from autograd import grad
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
         
    def C(self, beta: np.matrix, indx: np.matrix = None) -> np.matrix:
        """
            Returuns the value of the cost function at a new beta values
            Parameters:
                beta (np.matrix): features vector
        """
        return np.power((self.X[indx] @ beta - self.y[indx]), 2) / self.n

    def grad_C(self, beta: np.matrix, indx: np.matrix = None) -> np.matrix:
        """
            Returns the gradient of the function evaluated at a new beta values, 
            using the analytical expression. If no analytical expression is available,
            autograd will do the numerical approximation of the gradient.
            Parameters:
                beta (np.matrix): features vector
        """
        return (2 / self.n) * self.X[indx].T @ (self.X[indx] @ beta - self.y[indx])
        
    def grad_C_autograd(self, beta: np.matrix, indx: np.matrix = None) -> np.matrix:
        """
            Returns the gradient of the function evaluated at a new beta values,
            using autograd module.
            Parameters:
                beta (np.matrix): features vector
        """
        if not self.calculated_grad:    
            temp_grad = grad(self.C)
            self.calculated_grad = True
        return temp_grad(beta)
