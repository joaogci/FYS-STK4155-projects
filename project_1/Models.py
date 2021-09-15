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
    @param pseudo_inverse {boolean} Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv; ignored when from_svd is True
    @param verbose {boolean} If true, will print intermediate results to the console
    @returns Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
    """
    def ols(self, X, y, pseudo_inverse = True, verbose = False):
        
        # Compute beta from the matrix inverse or pseudo inverse
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

    

    """
    Ordinary Least Squares from SVD, with optional beta computation (slower)
    @param X {matrix} Design matrix
    @param y {vector} Training data
    @param include_beta {boolean} Whether to compute and return beta; slower than simply doing the prediction by itself
    @param verbose {boolean} If true, will print intermediate results to the console
    @returns Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
    """
    def ols_svd(self, X, y, include_beta = False, verbose = False):
        # Compute SVD of design matrix
        u, sigma, vT = np.linalg.svd(X, full_matrices=False)

        # Compute prediction
        pred = u @ u.T @ y

        # Compute beta (slow)
        beta = None
        if include_beta:
            diag = np.zeros((len(u), len(vT)))
            diag = np.diag(sigma)
            beta = vT.T @ np.linalg.inv(diag) @ u.T @ y

        if verbose:
            print("Ordinary least squares from SVD with design matrix:\n", X)
            print("Singular values:\n", sigma)
            if include_beta:
                print("Beta:\n", beta)
            else:
                print("Beta:\n(skipped for performance)")
            print("Prediction:\n", pred, "\n")
        
        # Return results
        return pred, beta


