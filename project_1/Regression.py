class Regression():
    """
        Parent class for Models and Resampling
        This class should include MSE, R2, setting up the Vandermonte matrix, and scaling
    """
    ...
    
    # --- Error functions ---

    def R2(y_data, y_model):
        return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
    
    def MSE(y_data,y_model):
        n = np.size(y_model)
        return np.sum((y_data-y_model)**2)/n
