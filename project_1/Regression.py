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
        self.x1 = np.ravel(x1)
        self.y = np.ravel(y)
        self.verbose = verbose
    
    @classmethod
    def from_2D(cls, x1, x2, y, verbose = False):
        """
            Constructor for 2D regression with additional x2
        """
        instance = cls(x1, y, verbose)
        instance.x2 = np.ravel(x2)
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

        self.X = np.ones((len(self.x1),degree+1)) # First column of design matrix is 1

        for i in range(1,degree+1): # First column is 1, so we skip it
            self.X[:,i] = self.x1**i
                
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

        self.X = np.ones((len(self.x1), int((degree + 1) * (degree + 2) / 2))) # First column of design matrix is 1

        for i in range(1, degree + 1): # First column is 1, so we skip it
            q = int(i * (i + 1) / 2)      # 1 + 2 + ... + i
            for k in range(i + 1):
                self.X[:, q + k] = (self.x1 ** (i - k)) * (self.x2 ** k)
        
    def tt_split(self, split = 0.25, seed = 0):
        """
            Splits the design matrix and data into two sets of data; testing and training
            Parameters:
                split {float}: Between 0 and 1, the fraction of data to use for testing
                seed {int}: Seed to use for the random generator
            Returns:
                {matrix, matrix, vector, vector} Split version of the data/design matrix
        """

        # Check inputs
        assert(self.X.shape[0] == self.y.shape[0])#, "".join(('ERROR: tt_split was given inputs of different sizes! Expects n_row(X) == len(y), given n_row(X) =', str(self.X.shape[0]), ' len(y) =', str(self.y.shape[0]), '!!')))
               
        # Init random number generator
        rng = np.random.default_rng(seed=seed)

        # Split the data into train and test sets
        split_size = int(self.X.shape[0] * split)
        
        perm = rng.permuted(np.arange(0, self.X.shape[0]))
        perm_X = self.X[perm, :]
        perm_y = self.y[perm]
        
        self.X_train = perm_X[0:split_size, :]
        self.y_train = perm_y[0:split_size]
        self.X_test = perm_X[split_size:, :]
        self.y_test = perm_y[split_size:]
            
        # return self.X_train, self.X_test, self.y_train, self.y_test
    
    def ensure_split(self):
        """
            Ensures tt_split has been called on the instance
            If not, runs tt_split before returning with default params

            Returns:
                (boolean): Whether tt_split had already been called on the instance
        """
        if not hasattr(self, 'X_train'):
            self.tt_split()
            return False
        return True
    
    # Skip the first value in the scaling since we disregard the intercept when scaling
    
    def standard_scaler(self):
        self.ensure_split()
         
        for i in range(1, self.X_train.shape[1]):
            mean_value = np.mean(self.X_train[:, i])
            standard_deviation = np.std(self.X_train[:, i])
            
            self.X_train[:, i] = (self.X_train[:, i] - mean_value) / standard_deviation
            self.X_test[:, i] = (self.X_test[:, i] - mean_value) / standard_deviation 
    
    def min_max_scaler(self):
        self.ensure_split()
        
        for i in range(1, self.X_train.shape[1]):
            x_min = np.min(self.X_train[:, i])
            x_max = np.max(self.X_train[:, i])
            
            self.X_train[:, i] = (self.X_train[:, i] - x_min) / (x_max - x_min)
            self.X_test[:, i] = (self.X_test[:, i] - x_min) / (x_max - x_min)
            
    def robust_scaler(self):
        self.ensure_split()
        
        for i in range(1, self.X_train.shape[1]):
            median = np.median(self.X_train[:, i])
            inter_quantile_range = np.percentile(self.X_train[:, i], 75) - np.percentile(self.X_train[:, i], 25)
            
            self.X_train[:, i] = (self.X_train[:, i] - median) / inter_quantile_range
            self.X_test[:, i] = (self.X_test[:, i] - median) / inter_quantile_range
