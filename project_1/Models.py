from Regression import Regression
import numpy as np

class Models(Regression):
    """
        Linear models class
    """

    """
    Initialises a Models instance with some optional settings
    @param verbose {boolean} If true, will print intermediate results to the console as applicable
    """
    def __init__(self, verbose = False):
        self.verbose = verbose

    
    # --- Error functions ---

    """
    Compute R2 score
    @param y_data {vector} Input data points to compare against
    @param y_model {vector} Predicted data
    @returns {float} The computed R2 score, which hopefully approaches 1
    """
    def r2(self, y_data, y_model):
        return 1 - np.sum(np.power(y_data - y_model, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2))
    
    """
    Compute Mean Squared Error
    @param y_data {vector} Input data points to compare against
    @param y_model {vector} Predicted data
    @returns {float} The computed Mean Squared Error, which hopefully approaches 0
    """
    def mse(self, y_data, y_model):
        return np.sum(np.power(y_data-y_model, 2)) / np.size(y_model)
    


    """
    Ordinary Least Squares
    @param X {matrix} Design matrix
    @param y {vector} Training data
    @param pseudo_inverse {boolean} Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv; ignored when from_svd is True
    @returns {vector, matrix} Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
    """
    def ols(self, X, y, pseudo_inverse = True):
        
        # Compute beta from the matrix inverse or pseudo inverse
        if pseudo_inverse: # Use numpy.linalg.pinv (uses SVD)
            beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        else: # Use true matrix inverse (may be ill-conditioned)
            beta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Compute prediction
        pred = X @ beta

        if self.verbose:
            print("Ordinary least squares with design matrix:\n", X)
            print("Regression parameters:", beta[:,0])
            print("MSE:", self.mse(y, pred))
            print("R2 score:", self.r2(y, pred))
            print("Training data prediction:", pred, "\n")

        # Return results
        return pred, beta

    

    """
    Ordinary Least Squares from SVD, with optional beta computation (slower)
    @param X {matrix} Design matrix
    @param y {vector} Training data
    @param include_beta {boolean} Whether to compute and return beta; slower than simply doing the prediction by itself
    @returns {vector, matrix|None} Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
    """
    def ols_svd(self, X, y, include_beta = False):
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

        if self.verbose:
            print("Ordinary least squares from SVD with design matrix:\n", X)
            print("Singular values:\n", sigma)
            if include_beta:
                print("Beta:\n", beta)
            else:
                print("Beta:\n(skipped for performance)")
            print("MSE:", self.mse(y, pred))
            print("R2 score:", self.r2(y, pred))
            print("Prediction:\n", pred, "\n")
        
        # Return results
        return pred, beta
