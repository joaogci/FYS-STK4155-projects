
from PredictionSource import PredictionSource
from InputSet import InputSet
import numpy as np
from DataGenerator import DataGenerator
from Splitter import Splitter
from Model import Model
from PostProcess import PostProcess
from Scaler import Scaler
import copy

class Solver:
    """
        Component-based solver implementation
        Components can be added to a solver instance to set up a model, a data generator, a scaler, etc.
    """


    def __init__(self, degree: int, fit_intercept: bool = False, data_generator: DataGenerator = None, splitter: Splitter = None, scaler: Scaler = None, models = list(), post_processes = list(), seed: int = 0):
        """
            Default Solver constructor
            Parameters can be used to set up components on the Solver in a non-verbose way
        """
        self._degree = degree
        self.fit_intercept = fit_intercept
        self._data_generator = data_generator
        self._splitter = splitter
        self._models = copy.deepcopy(models)
        self._post_processes = copy.deepcopy(post_processes)
        self._rng = np.random.default_rng(np.random.MT19937(seed))
        self._scaler = scaler
        self._data = None

        # Generate the data
        if self._data_generator != None:
            self._data = self._data_generator.generate(self._rng) 
    
    def set_data_generator(self, data_generator: DataGenerator):
        """
            Sets the data generator on the solver instance
            Parameters:
                data_generator (DataGenerator): The data generator to use
        """
        self._data_generator = data_generator
        
        # Generate the data
        self._data = self._data_generator.generate(self._rng)

    def set_data(self, data):
        """
            Sets the data to use directly without coming from a DataGenerator instance
            Parameters:
                data (tuple): 2- or 3- element tuple with column np matrices for X, Y and Z (X, Y for 2D)
        """
        self._data = data

    def get_data(self):
        """
            Returns the input data, either set directly through set_data or from the DataGenerator instance
        """
        if self._data is None:
            print('Error: Cannot obtain data from Solver instance before a call to either set_data_generator or set_data!')
            return None
        return self._data
    
    def set_splitter(self, splitter: Splitter):
        """
            Sets the splitter on the solver instance
            Parameters:
                splitter (Splitter): The splitter to use
        """
        self._splitter = splitter
    
    def add_model(self, model: Model):
        """
            Sets the model on the solver instance
            Parameters:
                model (Model): The model to use to predict the data
        """
        self._models.append(model)
        
    def set_scaler(self, scaler: Scaler):
        """
            Sets the scaler on the solver instance
            Parameters:
                scaler (Scaler): The scaler to use to scale the data
        """
        self._scaler = scaler

    def add_post_process(self, post_process: PostProcess):
        """
            Adds a post-process pass to the solver instance
            Parameters:
                post_process (PostProcess): The post process pass to append and run once predictions are made
        """
        self._post_processes.append(post_process)


    def _design_matrix(self, x1: np.matrix, x2: np.matrix) -> np.matrix:
        """
            Create the design matrix in the form of a Vandermonde matrix for one or two
            dimensional data set. The matrix is of the form

            [[1 x_1 x_1^2 ... x_1^(degree)]
             ...
             [1 x_n x_n^2 ... x_n^(degree)]]

            in 1D, and

            [[1 x_1 y_1 x_1^2 x_1y_1 y_1^2 ... y_1^(degree)]
             ...
             [1 x_n y_n x_n^2 x_ny_n y_n^2 ... y_n^(degree)]]

            in 2D.
            
            Parameters: 
                x1 (numpy array): The input data points on the X axis
                x2 (numpy array): The input data points on the Y axis, or None if generating 1D set
            
            Returns: 
                (numpy array): (n x p) dimensional matrix, 
                where n is number of datapoints and p is the degree 
                plus 1 (1-variable input) or p = degree*(degree + 1)/2 (2-variable input)
        """

        if x2 is None: # 1-variable Vandermonde matrix
            X = np.ones((len(x1), self._degree + 1)) # First column of design matrix is 1

            for i in range(1, self._degree+1): # First column is 1, so we skip it
                X[:, i] = x1.T ** i
        
        else: # 2-variable Vandermonde matrix
            X = np.ones((len(x1), int((self._degree + 1) * (self._degree + 2) / 2))) # First column of design matrix is 1
            # The number of features are 1 + 2 + ... + (degree+1) = (degree+1)*(degree+2)/2
            for i in range(1, self._degree + 1): # First column is 1, so we skip it
                q = int(i * (i + 1) / 2) # 1 + 2 + ... + i
                for k in range(i + 1):
                    X[:, q + k] = (x1[:, 0] ** (i - k)) * (x2[:, 0] ** k)
        
        # Return design matrix
        return X


    def run(self):
        """
            Runs the data generator, model, and other attached components to create necessary prediction(s)
        """
        
        # Check setup
        if self._data is None:
            print('Error: no data or data generator has been defined on the solver!')
            return
        if len(self._models) <= 0:
            print('Error: no model has been defined on the solver!')
            return
        
        # Create initial design matrix
        X = self._design_matrix(self._data[0], self._data[1] if len(self._data) > 2 else None)

        # Split data optionally into different InputSets
        # The different sets created here will each be a destination set for a prediction from one source set
        # For example, if the data is split into Test and Train sets, prediction_sources will be
        #           Train => [ Test, Train, Full ]
        #   as in, the Training set will be used to generate the prediction for all 3 sets (Full is optional here,
        #   but we compute it anyways to display the MSE/R2)
        if self._splitter != None:
            sets = self._splitter.split(X, self._data[-1])
            prediction_sources = self._splitter.prediction_sources()
        else:
            sets = {'full': InputSet(name='Full', X=X, y=self._data[-1])}
            prediction_sources = [PredictionSource(name='Full', src_set='full', dst_sets=['full'])] # Only use the full (and only) set as prediction source

        # Scale data optionally
        if self._scaler != None:
            for source in prediction_sources:
                # Take source set from the prediction source and scale design matrices in all destination sets
                self._scaler.prepare(sets[source.src_set].X)
                for dst in source.dst_sets:
                    sets[dst].X_scaled = self._scaler.scale(sets[dst].X)
                # Take source set from the prediction source and scale outcome matrices in all destination sets
                self._scaler.prepare(sets[source.src_set].y)
                for dst in source.dst_sets:
                    sets[dst].y_scaled = self._scaler.scale(sets[dst].y)
        
        # Set first column of design matrices to 0s instead of 1s if fit_intercept is False
        if not self.fit_intercept: 
            for key in sets.keys():
                sets[key].X[:, 0] = 0
                if sets[key].X_scaled is not None:
                    sets[key].X_scaled[:, 0] = 0
        
        # Init models and get beta for each model and for each source set, to make predictions out of in the destination sets
        # e.g. if 3 models are defined (OLS, Ridge(lmb=0.1) and Ridge(lmb=0.01)), and 2 prediction sources (train_0->test_0 and train_1->test_1)
        #       6 betas would be computed; one from train_0 using OLS, one from train_0 using Ridge(lmb=0.1), one from train_0 using Ridge(lmb=0.01), etc.
        #       and then from these, predictions would be made for test_0 using OLS, test_0 using Ridge(0.1), ...., test_1 using Ridge(0.01)
        for source in prediction_sources:
            src_set = sets[source.src_set] # The InputSet containing the X and y to use to interpolate for the selected prediction source
            for model in self._models:
                beta = model.interpolate(src_set.get_src_design_mat(), src_set.get_src_y()) # X and y may be scaled
                for dst in source.dst_sets: # Each of the destination InputSets that should be predicted from the source (which generally includes the source)
                    dst_set = sets[dst]
                    # Set the beta on the model and compute the prediction
                    dst_set.set_beta(model.name, beta)
        
        # No need to make predictions ahead of time; each relevant set has been assigned a beta value, so predictions can be made for a particular model with InputSet::get_prediction
        
        # Run post-processes on original data + full prediction
        for process in self._post_processes:
            process.run(self._data, sets, prediction_sources, self._models, self._degree)
