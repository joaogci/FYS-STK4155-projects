
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
