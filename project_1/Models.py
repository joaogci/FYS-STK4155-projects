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

    def print_error_estimates(self, name = None):
        """
            Displays the R2 score and MSE for the predictions made by the model

            Parameters:
                name (string|None): If provided, will display in brackets after the name of the different scores
        """
        prefix = "" if name == None else " (" + name + ")"
        print("MSE" + prefix + ":", self.mse(self.y, self.prediction))
        print("R2 score" + prefix + ":", self.r2(self.y, self.prediction))
        if hasattr(self, 'X_train'):
            print("Training MSE" + prefix + ":", self.mse(self.y_train, self.prediction_train))
            print("Training R2 score" + prefix + ":", self.r2(self.y_train, self.prediction_train))
            print("Test MSE" + prefix + ":", self.mse(self.y_test, self.prediction_test))
            print("Test R2 score" + prefix + ":", self.r2(self.y_test, self.prediction_test))
        print('')

    

    def ols(self, pseudo_inverse = True):
        """
            Ordinary Least Squares
            
            Parameters:
                pseudo_inverse (boolean): Whether to use numpy.linalg.pinv (SVD-based) instead of numpy.linalg.inv
        """

        # Pick the design matrix and data set - either the full set if tt_split hasn't been called,
        # or X_train and y_train
        X = self.X
        y = self.y
        training = False
        if hasattr(self, 'X_train'):
            X = self.X_train
            y = self.y_train
            training = True
        
        # Compute beta from the matrix inverse or pseudo inverse
        if pseudo_inverse: # Use numpy.linalg.pinv (uses SVD)
            self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        else: # Use true matrix inverse (may be ill-conditioned)
            self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Compute predictions
        self.prediction = self.X @ self.beta
        if training:
            self.prediction_train = self.X_train @ self.beta
            self.prediction_test = self.X_test @ self.beta

        if self.verbose:
            print("Ordinary least squares with design matrix:\n", self.X)
            print("Regression parameters:", self.beta[:,0])
            print("MSE:", self.mse(self.y, self.prediction))
            print("R2 score:", self.r2(self.y, self.prediction))
            print("Prediction:", self.prediction, "\n")


    def ols_svd(self, compute_beta = False):
        """
            Ordinary Least Squares from SVD, with optional beta computation (slower)
        
            Parameters:
                compute_beta (boolean): Whether to compute and return beta; slower than simply doing the prediction by itself
        """

        # Pick the design matrix and data set - either the full set if tt_split hasn't been called,
        # or X_train and y_train
        X = self.X
        y = self.y
        training = False
        if hasattr(self, 'X_train'):
            X = self.X_train
            y = self.y_train
            training = True
        
        # Compute SVD of design matrix
        u, sigma, vT = np.linalg.svd(X, full_matrices=False)

        # Compute prediction
        self.prediction = u @ u.T @ self.y
        if training:
            self.prediction_train = u @ u.T @ self.y_train
            self.prediction_test = u @ u.T @ self.y_test

        # Compute beta (slow)
        self.beta = None
        if compute_beta:
            diag = np.zeros((len(u), len(vT)))
            diag = np.diag(sigma)
            self.beta = vT.T @ np.linalg.inv(diag) @ u.T @ y

        if self.verbose:
            print("Ordinary least squares from SVD with design matrix:\n", X)
            print("Singular values:\n", sigma)
            if compute_beta:
                print("Beta:\n", self.beta)
            else:
                print("Beta:\n(skipped for performance)")
            print("MSE:", self.mse(self.y, self.prediction))
            print("R2 score:", self.r2(self.y, self.prediction))
            print("Prediction:\n", self.prediction, "\n")
