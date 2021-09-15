import numpy as np

class Regression:
    """
        Parent class for Models and Resampling
        This class should include MSE, R2, setting up the Vandermonte matrix, and scaling
    """
    def design_matrix(self, x,degree):
        # Flatten measure points if they are not 1 dim
        if len(x.shape) > 1:
            x = np.ravel(x)

        design_mat = np.ones((len(x),degree+1)) # First column of design matrix is 1

        for i in range(1,degree+1): # First column is 1, so we skip it
            design_mat[:,i] = x**i

        return design_mat
    
    def design_matrix_2D(self, x,y,degree):
        # Flatten measure points if they are not 1 dim
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        len_beta = int((degree+1)*(degree+2)/2)	# Number of elements in beta
        design_mat = np.ones((len(x),len_beta)) # First column of design matrix is 1

        for i in range(1,degree+1): # First column is 1, so we skip it
            q = int(i*(i+1)/2)      # 1 + 2 + ... + i
            for k in range(i+1):
                design_mat[:,q+k] = (x**(i-k))*(y**k)

        return design_mat

    

    """
    Splits the design matrix and data into two sets of data; testing and training
    @param X {matrix} The matrix to split
    @param y {vector} The data to split
    @param split {float} Between 0 and 1, the fraction of data to use for testing
    @param seed {int} Seed to use for the random generator
    @returns {matrix, matrix, vector, vector}
    """
    def tt_split(self, X, y, split = 0.25, seed = 0):

        # Check inputs
        if len(X) != len(y):
            print('ERROR: tt_split was given inputs of different sizes! Expects len(X) == len(y), given len(X) =', len(X), ', len(y) =', len(y), '!!')
            return None, None, None, None

        rng = np.random.default_rng(seed=seed)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(0, len(X)):
            r = rng.normal()
            if r < split:
                


