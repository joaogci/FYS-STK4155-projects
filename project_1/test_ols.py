
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from FrankeGenerator import FrankeGenerator
from OLSModel import OLSModel
from RidgeModel import RidgeModel
from LassoModel import LassoModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess
from TrainTestSplitter import TrainTestSplitter
from StandardScaler import StandardScaler
from ExponentialGenerator import ExponentialGenerator
from MinMaxScaler import MinMaxScaler
from RobustScaler import RobustScaler

show_franke = False
degree = 5

solver = Solver(degree, fit_intercept=True, seed=2)
if show_franke:
    solver.set_data_generator(FrankeGenerator(0, 1, 100))
else:
    solver.set_data_generator(PolynomialGenerator(count=100, min_x=-15, max_x=20, noise=0.01))
solver.add_model(OLSModel(0.01))
solver.set_scaler(MinMaxScaler())
solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())
solver.run()
