import numpy as np

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
            design_matrix[:, q + k] = (X[:, 0] ** (i - k)) * (Y[:, 0] ** k)
    
    return design_matrix

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

def scale_mean(X_train: np.matrix, X_test: np.matrix, y_train: np.matrix, y_test: np.matrix):
    """
        Subtracts the mean value from input data
        
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
    
    mean_X = np.mean(X_train, axis=0)
    X_train_scaled = X_train - mean_X
    X_test_scaled = X_test - mean_X

    mean_y = np.mean(y_train)
    y_train_scaled = y_train - mean_y
    y_test_scaled = y_test - mean_y
    
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

def bias_squared(y_data: np.matrix, y_model: np.matrix):
    """
        Compute bias squared
        
        Parameters:
            y_data (numpy array) input data points to compare against
            y_model (numpy array) predicted data
            
        Returns:
            (float) the computed bias squared
    """
 
    return np.mean((y_data - np.mean(y_model))**2)

def variance(y_model: np.matrix):
    """ 
        Compute variance of given data
        
        Parameters:
            y_model (numpy array) predicted data
            
        Returns:
            (float) the computed variance
    """
    
    return np.mean((y_model - np.mean(y_model))**2)

def ols(X_train, y_train):
    """
        Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
        Predictions will be based off OLS for this model
        
        Parameters:
            X_train (numpy matrix) design matrix for training
            y_train (numpy array) target data for traning
        
        Returns:
            (numpy array) optimal coefficients (beta) of the linear regression
    """ 
    
    return np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train







