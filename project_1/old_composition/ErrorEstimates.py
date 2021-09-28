
import numpy as np

"""
    Common error estimate functions
"""
    

def r2(y_data: np.matrix, y_model: np.matrix) -> float:
    """
        Compute R2 score
        
        Parameters:
            y_data (vector) Input data points to compare against
            y_model (vector) Predicted data
            
        Returns:
            (float) The computed R2 score, which hopefully approaches 1
    """
    return 1 - np.sum(np.power(y_data - y_model, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2))

def mse(y_data: np.matrix, y_model: np.matrix) -> float:
    """
        Compute Mean Squared Error
        
        Parameters:
            y_data (vector): Input data points to compare against
            y_model (vector): Predicted data
        
        Returns:
            (float) The computed Mean Squared Error, which hopefully approaches 0
    """
    return np.sum(np.power(y_data-y_model, 2)) / np.size(y_model)

def beta_conf_intervals(X: np.matrix) -> np.matrix:
    """
        Compute confidence intervals for beta
        
        Parameters:
            X (matrix): Design matrix
        
        Returns:
            (vector) confidence interval for every beta
    """ 
    temp = np.linalg.pinv(X.T @ X) 
    var = np.zeros(temp.shape[0])
    
    for i in range(var.size):
        var[i] = temp[i, i]
    
    return np.sqrt(var)
