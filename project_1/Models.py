import Regression
import numpy as np

class Model(Regression):
    """
        Linear models class
    """
    

    # Ordinary Least Squares
    # @param X {matrix} Design matrix
    # @param y {vector} Training data
    # @param pseudoInverse {boolean} Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv
    # @param verbose {boolean} If true, will print intermediate results to the console
    # @returns Returns the training prediction and the regression parameters vector (beta)
    def ols(X, y, pseudoInverse = False, verbose = False):
        
        # Find beta
        if pseudoInverse:   # Compute beta from the SVD (numpy.linalg.inv)
            beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        else:               # Compute beta from the matrix inverse (may be prone to errors)
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
        
        if verbose:
            print("Ordinary least squares with design matrix:", X, "\nRegression parameters:", beta, "\nTraining data prediction:", beta @ y, "\n")

        # Return results
        return beta @ y, beta

    
