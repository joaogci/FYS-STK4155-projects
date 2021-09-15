from Regression import Regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Models(Regression):
    """
        Holds various models that can be used to fit the data to a polynomial
    """


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


    def ols_svd(self):
        """
            Ordinary Least Squares from SVD, with optional beta computation (slower)
        
            Parameters:
                compute_beta (boolean): Whether to compute and return beta; slower than simply doing the prediction by itself
        """
        
        # Compute SVD of design matrix
        u, sigma, vT = np.linalg.svd(self.X_train, full_matrices=False)
        
        self.prediction = u @ u.T @ self.y_train
        
        self.beta = np.multiply(vT.T, 1 / sigma) @ u.T @ self.y_train

        if self.verbose:
            print("Ordinary least squares from SVD with design matrix:\n", self.X_train)
            print("Singular values:\n", sigma)
            print("Beta:\n", self.beta)
            print("MSE:", self.mse(self.y_train, self.prediction))
            print("R2 score:", self.r2(self.y_train, self.prediction))
            print("Prediction:\n", self.prediction, "\n")

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

    def plot(self, add_data = True, name = None, colour = 'b'):
        """
            Adds the data (optionally) and prediction to the ongoing matplotlib

            Parameters:
                add_data (boolean): If true, the original data will be plotted alongside the prediction
                name (string|None): If provided, will display in brackets in the label
                colour (string): Colour to use for the prediction plot
        """
        
        if add_data:
            plt.plot(self.x1, self.y ,'k+', label='Input data')
        plt.plot(np.sort(self.x1, 0), np.sort(self.prediction, 0), colour + '-', label='Prediction'+('' if name == None else " (" + name + ")"))
        plt.legend()
        
    def plot_3D(self):
        Xm, Ym = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
        Zm = np.zeros((len(np.arange(0, 1, 0.01)), len(np.arange(0, 1, 0.01))))

        for i in range(0, 6):
            q = int(i * (i + 1) / 2)
            for k in range(i + 1):
                Zm += self.beta[q+k] * (Xm ** (i - k)) * (Ym ** k)
 
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(Xm, Ym, Zm, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        
        # ax.set_zlim(np.min(self.franke) - 0.3, np.max(self.franke) + 0.4)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)
