
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel
from RidgeModel import RidgeModel
from LassoModel import LassoModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess
from TrainTestSplitter import TrainTestSplitter
from StandardScaler import StandardScaler
from ExponentialGenerator import ExponentialGenerator

degree = 4

solver = Solver(degree, seed=0)

solver.set_data_generator(ExponentialGenerator(degree=degree, count=100, min_x=-1, max_x=1, noise=0.1))

solver.set_splitter(TrainTestSplitter())
solver.set_scaler(StandardScaler())

solver.add_model(OLSModel())
solver.add_model(RidgeModel(5))

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())
