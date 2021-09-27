# Project 1 - Fitting linear models to the Franke function

## Confusion (NEED TO FIX)

When scaling (without std), i.e., only "supposedly" removing the mean, the predictions are correct, but when plotting the actual polynomial using the beta-values, we get a wrong result.

## TODO: 
 * Resampling (bootstraping and cross-validation)
 * KFold spliting for CV

## Examples

Fitting a 1D 4th degree (randomized) polynomial using OLS:
```py
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel

solver = Solver(4) # 4th degree polynomial
# Set the data generator to a Polynomial Generator
solver.set_data_generator(PolynomialGenerator(degree=4, count=100, min_x=-5, max_x=5, noise=0.05))
# Set the model as OLS via SVD
solver.set_model(OLSModel())
# Run the solver
solver.run()
```

Fitting a 2D Franke Function sample using regular OLS to fit a 10th degree polynomial:
```py
from Solver import Solver
from FrankeGenerator import FrankeGenerator
from OLSModel import OLSModel

solver = Solver(10) # 10th degree polynomial
# Set the data generator to a Franke Generator (3D data)
solver.set_data_generator(FrankeGenerator(0, 1, 0.01, noise=0.01))
# Set the model to OLS
solver.set_model(OLSModel())
# Run the solver
solver.run()
```

Displaying a plot of the predictions made:
```py
from PlotPostProcess import PlotPostProcess

...
solver.add_post_process(PlotPostProcess())
...
solver.run()
```

Displaying the MSE and R2 scores for the different predictions:
```py
from ErrDisplayPostProcess import ErrDisplayPostProcess

...
solver.add_post_process(ErrDisplayPostProcess())
...
solver.run()
```

Splitting the data into training and testing sets:
```py
from TrainTestSplitter import TrainTestSplitter

...
solver.set_splitter(TrainTestSplitter(test_size=0.25))
...
solver.run()
```

## Classes

### Solver

A modular class to which different components can be added to fit different functions and data sets. Solver instances need at least a `DataGenerator` and a `Model` attached; optionally, a `Splitter` and several `PostProcess`es can also be attached to further split & analyze the predictions made.

### DataGenerator

`DataGenerator` child classes should implement the `generate` function to return a `tuple` with either 2 (2D) or 3 (3D) `np.matrix` elements; a `DataGenerator` instance can be attached to a `Solver` using `.set_data_generator()`.

Child classes: `PolynomialGenerator` (2D data), `FrankeGenerator` (3D data)

### Model

`Model` child classes should implement the `interpolate` method which takes a design matrix and data set and returns an estimator function which can later be fed other design matrices to fit a polynomial to them following the same trend as the original data set; a `Model` instance can be attached to a `Solver` using `.set_model()`.

Child classes: `OLSModel`, `RidgeModel`

### Splitter

`Splitter` child classes should implement the `split` method which takes a design matrix and data set and returns a collection of labeled data subsets; a `Splitter` instance can be attached to a `Solver` using `.set_splitter()`.

Child classes: `TrainTestSplitter`

### PostProcess

`PostProcess` child classes should implement the `run` method which takes all the original data, subsets and predictions and to run arbitrary processes on; e.g. plot the predictions (see `PlotPostProcess`), print the R2 score and Mean Squared Error (`ErrDisplayPostProcess`), etc. A `PostProcess` instance can be attached to a `Solver` using `.add_post_process()` - note that more than one `PostProcess` can be added to a single `Solver`.

Child classes: `PlotPostProcess`, `ErrDisplayPostProcess`

