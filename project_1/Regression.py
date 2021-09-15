import numpy as np

class Regression:
    """
        Parent class for Models and Resampling
        This class should include MSE, R2, setting up the Vandermonte matrix, and scaling
        
        Parameters for 1D regression: 
            x1 (vector): data points for fitting
            y (vector): data set for fitting
            verbose (boolean): If true, will print intermediate results to the console as applicable 
        
        Parameters for 2D regression:
            x1, x2 (vector): data points for fitting
            y (matrix): data set for fitting
            verbose (boolean): If true, will print intermediate results to the console as applicable 
    """
    
    def __init__(self, x1, y, verbose = False):
        self.x1 = x1
        self.y = y
        self.verbose = verbose
    
    @classmethod
    def from_2D(cls, x1, x2, y, verbose = False):
        """
            Constructor for 2D regression with additional x2
        """
        instance = cls(x1, y, verbose)
        instance.x2 = x2
        return instance

    def design_matrix(self, degree):
        """
            Create the design matrix in the form of a Vandermonde matrix for one 
            dimensional data set. The matrix is of the form

            [[1 x_1 x_1^2 ... x_1^(degree)]
             ...
             [1 x_n x_n^2 ... x_n^(degree)]]
            
            Parameters: 
                x (numpy array): The input data
                degree (int): The degree of the polynomial
            
            Returns: 
                design_mat (numpy array): (n x p) dimensional matrix, 
                where n is number of datapoints and p is the degree 
                pluss 1
        """

        # Flatten measure points if they are not 1 dim
        self.x1 = np.ravel(self.x1)

        design_mat = np.ones((len(self.x1),degree+1)) # First column of design matrix is 1

        for i in range(1,degree+1): # First column is 1, so we skip it
            design_mat[:,i] = self.x1**i
            
        self.X = design_mat
        return self.X
    
    def design_matrix_2D(self, degree):
        """
            Create the design matrix in the form of a Vandermonde matrix for two 
            dimensional data set. The matrix is of the form
            
            [[1 x_1 y_1 x_1^2 x_1y_1 y_1^2 ... y_1^(degree)]
             ...
             [1 x_n y_n x_n^2 x_ny_n y_n^2 ... y_n^(degree)]]
            
            Parameters: 
                x (numpy array): The input data
                y (numpy array): The input data
                degree (int): The degree of the polynomial
            
            Returns: 
                design_mat (numpy array): (n x p) dimensional matrix, 
                where n is number of datapoints and p is given by
                p = degree*(degree + 1)/2
        """

        # Flatten measure points if they are not 1 dim
        self.x1 = np.ravel(self.x1)
        self.x2 = np.ravel(self.x2)

        len_beta = int((degree + 1) * (degree + 2) / 2)	# Number of elements in beta
        design_mat = np.ones((len(self.x1), len_beta)) # First column of design matrix is 1

        for i in range(1, degree + 1): # First column is 1, so we skip it
            q = int(i * (i + 1) / 2)      # 1 + 2 + ... + i
            for k in range(i + 1):
                design_mat[:, q + k] = (self.x ** (i - k)) * (self.y ** k)
        
        self.X = design_mat
        return self.X

    def tt_split(self, X, y, split = 0.25, seed = 0):
        """
            Splits the design matrix and data into two sets of data; testing and training
            Parameters:
                X {matrix}: The matrix to split
                y {vector}: The data to split
                split {float}: Between 0 and 1, the fraction of data to use for testing
                seed {int}: Seed to use for the random generator
            Returns:
                {matrix, matrix, vector, vector} Split version of the data/design matrix
        """

        # Check inputs
        if len(X) != len(y):
            print('ERROR: tt_split was given inputs of different sizes! Expects len(X) == len(y), given len(X) =', len(X), ', len(y) =', len(y), '!!')
            return None, None, None, None

        # Init random number generator
        rng = np.random.default_rng(seed=seed)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = [], [], [], []
        for i in range(0, len(X)):
            r = rng.random()
            if r > split:
                X_train.append(X[i])
                y_train.append(y[i])
            else:
                X_test.append(X[i])
                y_test.append(y[i])

        # Return the split data as matrices
        return np.asmatrix(X_train), np.asmatrix(X_test), np.asmatrix(y_train), np.asmatrix(y_test)
    
    def standard_scaler(self, X_train, X_test): 
        
        for i in range(X_train.shape[1]):
            mean_value = np.mean(X_train[:, i])
            standard_deviation = np.std(X_train[:, i])
            
            X_train[:, i] = (X_train[:, i] - mean_value) / standard_deviation
            X_test[:, i] = (X_test[:, i] - mean_value) / standard_deviation 
        
        return X_train, X_test
    
    def min_max_scaler(self, X_train, X_test):
        
        for i in range(X_train.shape[1]):
            x_min = np.min(X_train[:, i])
            x_max = np.max(X_train[:, i])
            
            X_train[:, i] = (X_train[:, i] - x_min) / (x_max - x_min)
            X_test[:, i] = (X_test[:, i] - x_min) / (x_max - x_min)
            
        return X_train, X_test

    def robust_scaler(self, X_train, X_test):
       
       for i in range(X_train.shape[1]):
           median = np.median(X_train[:, i])
           inter_quantile_range = np.percentile(X_train[:, i], 75) - np.percentile(X_train[:, i], 25)
           
           X_train[:, i] = (X_train[:, i] - median) / inter_quantile_range
       
       return X_train, X_test 