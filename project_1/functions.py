import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample

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

def scale_mean_svd(X: np.matrix, y: np.matrix) -> tuple:
    """
        Subtracts the mean value and divides by the standard deviation

        Parameters:
            X (np.matrix): Design matrix
            y (np.matrix): Outcome

        Returns:
            (np.matrix): Scaled design matrix
            (np.matrix): Scaled outcome
    """
    compute = lambda m: (m - np.mean(m)) / np.std(m)
    return compute(X), compute(y)

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


class Regression():

    def __init__(self, max_degree: int, n: int, noise: float, rng: np.random.Generator, scale: bool = True, data: tuple = None):
        """
            Regression class
            
            Creates the design matrix, generates the data from the Franke function in a random uniform interval [0, 1).
            Also adds noise to the data, sampled over a normal distribution (N(0, noise))
            
            Parameters:
                max_degree (int): max polynomial degree to fit
                n (int): number of data points
                noise (float): variance for the noise
                rng (numpy Generator): random number generator
                scale (bool): wheter to scale or not the data
        """
        
        self.max_degree = max_degree
        self.rng = rng
        self.noise = noise
        self.data_points = n
        
        if data is None:
            self.x = rng.uniform(0, 1, (n, 1))
            self.y = rng.uniform(0, 1, (n, 1))
            
            self.z = franke_function(self.x, self.y)
            self.z += noise * rng.normal(0, 1, self.z.shape)
        else:
            self.x = data[0]
            self.y = data[1]
            self.z = data[2]
        
        self.X_max_deg = create_X_2D(max_degree, self.x, self.y)
        self.X_train_, self.X_test_, self.z_train, self.z_test = train_test_split(self.X_max_deg, self.z, test_size=0.25)
        
        self._scaled = False
        if scale:
            self.X_train_, self.X_test_, self.z_train, self.z_test = scale_mean(self.X_train_, self.X_test_, self.z_train, self.z_test)
            self._scaled = True

    def ordinary_least_squares(self, degree: int, scale: bool = True):
        """
            Ordrinary Least Squares function
        """
        
        if not self._scaled and scale:
            X_train_, X_test_, z_train, z_test = scale_mean(self.X_train_, self.X_test_, self.z_train, self.z_test)
            X_train = X_train_[:, :self._n_features(degree)]
            X_test = X_test_[:, :self._n_features(degree)]
        else:
            X_train = self.X_train_[:, :self._n_features(degree)]
            X_test = self.X_test_[:, :self._n_features(degree)]
            z_train = self.z_train
            z_test = self.z_test
        
        betas = ols(X_train, z_train)
        var_betas = self.noise**2 * np.diag(np.linalg.pinv(X_test.T @ X_test))
        
        z_pred = X_train @ betas
        z_tilde = X_test @ betas
        
        mse_train = mean_squared_error(z_train, z_pred)
        mse_test = mean_squared_error(z_test, z_tilde)
        r2_train = r2_score(z_train, z_pred)
        r2_test = r2_score(z_test, z_tilde)
        
        return mse_train, r2_train, mse_test, r2_test, betas, var_betas

    def bootstrap(self, degree: int, max_bootstrap_cycle: int, lmd: float = 0, alpha: float = 0):
        """
            Bootstrap function
            
            Parameters: 
                degree (int): polynomial degree to fit
                max_bootstrap_cycle (int): max bootstrap iterations
                lmd
        """

        # select wanted features
        X_train = self.X_train_[:, :self._n_features(degree)]
        X_test = self.X_test_[:, :self._n_features(degree)]
        
        z_tilde_all = np.zeros((self.z_test.shape[0], max_bootstrap_cycle))
        
        for bootstrap_cycle in range(max_bootstrap_cycle):
            print(f"n={self.data_points} | bootstrap cycle {bootstrap_cycle+1}/{max_bootstrap_cycle} with degree {degree}/{self.max_degree} and lmd={lmd if alpha == 0 else alpha}", end="\r")
            
            # split and scale the data
            X_train_resampled, z_train_resampled = resample(X_train, self.z_train)
            
            if alpha == 0:  # ridge and ols
                # fit OLS model to franke function
                betas = ols(X_train_resampled, z_train_resampled, lmd=lmd)
                # predictions
                z_tilde_all[:, bootstrap_cycle] = (X_test @ betas).reshape((self.z_test.shape[0], ))
            else:           # lasso with sklearn
                lasso = Lasso(alpha=alpha, fit_intercept=False, normalize=True,tol=1e-2, max_iter=1e8)
                lasso.fit(X_train_resampled, z_train_resampled)
                z_tilde_all[:, bootstrap_cycle] = lasso.predict(X_test).reshape((self.z_test.shape[0], ))

        # compute MSE, BIAS and VAR
        mse_test = mean_squared_error(self.z_test, z_tilde_all)
        bias = np.mean((self.z_test.reshape(self.z_test.shape[0], ) - np.mean(z_tilde_all, axis=1))**2)
        var = np.mean(np.var(z_tilde_all, axis=1))
        
        # to update print line
        print()
        return mse_test, bias, var
    
    def k_folds_cross_validation(self, degree: int, n_folds: int, lmd: float = 0, alpha: float = 0):
        """
            K Folds cross validation
        """
        
        print(f"n={self.data_points} | n_folds={n_folds} with degree {degree}/{self.max_degree} and lmd={lmd if alpha == 0 else alpha}")
        
        # select wanted features
        X = self.X_max_deg[:, :self._n_features(degree)]

        # initialize KFold object
        kfolds = KFold(n_splits=n_folds)
        
        # perform the cross-validation to estimate MSE
        scores_KFold = np.zeros(n_folds)

        i = 0
        for train_inds, test_inds in kfolds.split(X):
            # select k_folds data
            X_train = X[train_inds, :]
            z_train = self.z[train_inds]

            X_test = X[test_inds, :]
            z_test = self.z[test_inds]
            
            X_train, X_test, z_train, z_test = scale_mean(X_train, X_test, z_train, z_test)

            if alpha == 0:  # ridge and ols
                # model and prediction
                betas = ols(X_train, z_train, lmd=lmd)
                z_tilde = X_test @ betas
            else:           # lasso
                lasso = Lasso(alpha=alpha, fit_intercept=False, normalize=True, tol=1e-2, max_iter=1e8)
                lasso.fit(X_train, z_train)
                z_tilde = lasso.predict(X_test)

            scores_KFold[i] = mean_squared_error(z_test, z_tilde)
            i += 1
        
        return np.mean(scores_KFold)
    
    def _n_features(self, deg: int):
        """
            Returns the number of features for the design matrix when we fit a deg polynomial
            
            Parameters:
                deg (int): degree of fitting polynomial
                
            Returns:
                (int) number of features
        """
        return int((deg + 1) * (deg + 2) / 2)
    
        
        
        



