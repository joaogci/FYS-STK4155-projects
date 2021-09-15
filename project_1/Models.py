from Regression import Regression
import numpy as np

class Models(Regression):
    """
        Initialises a Models instance with some optional settings
    """

    def r2(self, y_data, y_model):
        """
            Compute R2 score
            
            Parameters:
                y_data (vector) Input data points to compare against
                y_model (vector) Predicted data
                
            Returns:
                (float) The computed R2 score, which hopefully approaches 1
        """
        
        return 1 - np.sum(np.power(y_data - y_model, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2))
    

    def mse(self, y_data, y_model):
        """
            Compute Mean Squared Error
            
            Parameters:
                y_data (vector): Input data points to compare against
                y_model (vector): Predicted data
            
            Returns:
                (float) The computed Mean Squared Error, which hopefully approaches 0
        """
        
        return np.sum(np.power(y_data-y_model, 2)) / np.size(y_model)
    


    def ols(self, X, y, pseudo_inverse = True):
        """
            Ordinary Least Squares
            
            Parameters:
                X (matrix): Design matrix
                y (vector): Training data
                pseudo_inverse (boolean): Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv; ignored when from_svd is True
            
            Returns:
                (vector, matrix) Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
        """
        
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


    def ols_svd(self, X, y, include_beta = False):
        """
            Ordinary Least Squares from SVD, with optional beta computation (slower)
        
            Parameters: 
                X (matrix): Design matrix
                y (vector): Training data
                include_beta (boolean): Whether to compute and return beta; slower than simply doing the prediction by itself

            Returns:
                (vector, matrix|None) Returns the training prediction and the regression parameters vector (beta) (only returned if from_svd is False)
        """
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
