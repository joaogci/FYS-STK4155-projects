from Regression import Regression
import numpy as np

class Models(Regression):
    """
        Linear models class
    """
    
    """
    Ordinary Least Squares
    @param X {matrix} Design matrix
    @param y {vector} Training data
    @param from_svd {boolean} If true, will 
    @param pseudo_inverse {boolean} Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv; ignored when from_svd is True
    @param verbose {boolean} If true, will print intermediate results to the console
    @returns Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
    """
    def ols(self, X, y, use_svd = False, pseudo_inverse = True, verbose = False):
        
        # Find beta
        if pseudo_inverse:   # Compute prediction from the SVD 
            # Compute SVD of design matrix
            u, _, __ = np.linalg.svd(X, full_matrices=False)

            if verbose:
                print("Ordinary least squares from SVD with design matrix:\n", X)
                print("SVD found U:\n", u)
                print("Prediction:\n", u @ u.T @ y, "\n")

            # Return prediction only (no beta when done thru SVD)
            return u @ u.T @ y, None

        else: # Compute beta from the matrix inverse or pseudo inverse
            if pseudo_inverse: # Use numpy.linalg.pinv (uses SVD)
                beta = np.linalg.pinv(X.T @ X) @ X.T @ y
            else: # Use true matrix inverse (may be ill-conditioned)
                beta = np.linalg.inv(X.T @ X) @ X.T @ y

            if verbose:
                print("Ordinary least squares with design matrix:\n", X)
                print("Regression parameters:", beta[:,0])
                print("Training data prediction:", beta @ y, "\n")

            # Return results
            return X @ beta, beta

