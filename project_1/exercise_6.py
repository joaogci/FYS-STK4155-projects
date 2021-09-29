
from terrain import load_terrain, TERRAIN_1, TERRAIN_2
from functions import create_X_2D, ols
from plots import plot_prediction_3D



# constants
degree = 20
terrain_set = TERRAIN_1


# Load data set
x, y, z = load_terrain(terrain_set, downsample=0.5, rng=None, plot=True, show_plot=False)

# Design matrix
X = create_X_2D(degree, x, y)

# OLS
beta = ols(X, z)
z_pred = X @ beta

# Plot prediction
plot_prediction_3D(beta, degree, 0, 1, 0, 1, name=TERRAIN_1+' prediction')
