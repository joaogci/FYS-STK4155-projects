
import numpy as np

def franke_function(X, Y):
    """
        Franke Function
        
        Parameters: 
            X (numpy array): mesh x data
            Y (numpy array): mesh y data
            
        Returns: 
            (numpy array): franke functon over X and Y
    """
    
    term1 = 0.75*np.exp(-(0.25*(9*X-2)**2) - 0.25*((9*Y-2)**2))
    term2 = 0.75*np.exp(-((9*X+1)**2)/49.0 - 0.1*(9*Y+1))
    term3 = 0.5*np.exp(-(9*X-7)**2/4.0 - 0.25*((9*Y-3)**2))
    term4 = -0.2*np.exp(-(9*X-4)**2 - (9*Y-7)**2)
    return term1 + term2 + term3 + term4

def create_X_2D(degree: int, X: np.matrix, Y: np.matrix):
    """
        Create the design matrix in the form of a Vandermonde matrix for one or two
        dimensional data set. The matrix is of the form

        [[1 x_1 y_1 x_1^2 x_1y_1 y_1^2 ... y_1^(degree)]
            ...
            [1 x_n y_n x_n^2 x_ny_n y_n^2 ... y_n^(degree)]]

        in 2D.
        
        Parameters: 
            degree (int): degree of the polynomial to fit (p)
            X (numpy array): The input data points on the X axis (n)
            Y (numpy array): The input data points on the Y axis (n)
        
        Returns: 
            (numpy array): (n x p) dimensional matrix, 
            where n is number of datapoints and p is the degree 
            plus p = degree*(degree + 1)/2 (2-variable input)
    """
    
    design_matrix = np.ones((len(X), int((degree + 1) * (degree + 2) / 2))) # First column of design matrix is 1
    
    # The number of features are 1 + 2 + ... + (degree+1) = (degree+1)*(degree+2)/2
    for i in range(1, degree + 1): # First column is 1, so we skip it
        q = int(i * (i + 1) / 2) # 1 + 2 + ... + i
        
        for k in range(i + 1):
            design_matrix[:, q + k] = (X ** (i - k)) * (Y ** k)
    
    return design_matrix

def n_features(deg):
    """
        Returns the number of features for a 2D polynomial fit
    """
    return int((deg + 1) * (deg + 2) / 2)

def scale_mean_std(X_train: np.matrix, X_test: np.matrix, y_train: np.matrix, y_test: np.matrix) -> tuple:
    """
        Subtracts the mean value and divides by the standard deviation

        Parameters:
            X_train (numpy matrix) training design matrix
            X_test (numpy matrix) testing design matrix
            y_train (numpy array) training target
            y_test (numpy array) testing target
            
        Returns:
            (numpy matrix) training design matrix scaled
            (numpy matrix) testing design matrix scaled
            (numpy array) training target scaled
            (numpy array) testing target scaled
    """
    
    X_train[0, :] = 0

    mean_X = np.mean(X_train, axis=0)
    std_X = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X) / std_X
    X_test_scaled = (X_test - mean_X) / std_X

    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    y_train_scaled = (y_train - mean_y) / std_y
    y_test_scaled = (y_test - mean_y) / std_y
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

def mean_squared_error(y_data: np.matrix, y_model: np.matrix):
    """
        Compute Mean Squared Error
        
        Parameters:
            y_data (numpy array): input data points to compare against
            y_model (numpy array): predicted data
        
        Returns:
            (float) the computed Mean Squared Error
    """
    
    return np.mean((y_data - y_model)**2)

def r2_score(y_data: np.matrix, y_model: np.matrix):
    """
        Compute R2 score
        
        Parameters:
            y_data (numpy array) input data points to compare against
            y_model (numpy array) predicted data
            
        Returns:
            (float) the computed R2 score
    """
    
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def ols(X: np.matrix, y: np.matrix, lmd: float = 0) -> np.matrix:
    """
        Given a design matrix and a data set, returns the beta predictor array to be used to make predictions
        using OLS or Ridge regression with SVD pseudo-inverse

        Parameters:
            X (numpy matrix): Design matrix
            y (numpy array): Target data
            lmd (float): lmd value for Ridge (if lmd == 0, then OLS)
        
        Returns:
            (numpy array): Beta coefficients
    """

    return np.linalg.pinv(X.T @ X + lmd * np.eye(X.shape[1])) @ X.T @ y

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def plot_prediction_3D(beta: np.matrix, degree: int, min_x: float = 0, max_x: float = 1, min_y: float = 0, max_y: float = 1, name: str = 'Prediction', display_steps: int = 100, show: bool = True, save_fig: bool = False) -> None:
    """
        Given a beta feature matrix of a certain degree, plots the estimate of a 2D function over a certain domain

        Parameters:
            beta (np.matrix): Feature estimate given by the model
            degree (int): Polynomial degree
            min_x (float): Start of the domain to show on X
            max_x (float): End of the domain to show on X
            min_y (float): Start of the domain to show on Y
            max_y (float): End of the domain to show on Y
            name (str): Name of the plot to display
            display_steps (int): Number of points to show on X and Y; the overall number of data points computed is display_steps^2
            show (bool): Whether to show() the plot at the end
    """

    # Show prediction as a smooth plot
    fig = plt.figure(name)
    ax = plt.axes(projection='3d')

    # Generate linspaced meshgrid to show predictions at smooth points
    xm_display, ym_display = np.meshgrid(np.linspace(min_x, max_x, display_steps), np.linspace(min_y, max_y, display_steps))
    zm_display = np.zeros((display_steps, display_steps))
    betaIdx = 0
    for i in range(degree + 1):
        for k in range(i + 1):
            zm_display += beta[betaIdx] * (xm_display ** (i - k)) * (ym_display ** k)
            betaIdx += 1
    surf = ax.plot_surface(xm_display, ym_display, zm_display, cmap=cm.gray, linewidth=0, antialiased=True)

    # Plot surface
    ax.set_zlim(np.min(zm_display) - 1, np.max(zm_display) + 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")   
    plt.title(name)

    if show:
        plt.show()
    if save_fig: 
        plt.savefig("./images/ex6_pred_OLS.pdf")

