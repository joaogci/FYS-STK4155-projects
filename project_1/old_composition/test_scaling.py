
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
from MinMaxScaler import MinMaxScaler
from RobustScaler import RobustScaler

degree = 5

solver = Solver(degree, seed=0)

solver.set_data_generator(ExponentialGenerator(degree=degree, count=100, min_x=-1, max_x=1, noise=0.1))

solver.set_splitter(TrainTestSplitter())
solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.add_model(RidgeModel(lmd=0.5))

print("Without Scaling: ")
solver.run()

print("With StandardScaler with std")
solver.set_scaler(StandardScaler())
solver.run()

print("With StandardScaler without std")
solver.set_scaler(StandardScaler(with_std=False))
solver.run()

print("With MinMaxScaler")
solver.set_scaler(MinMaxScaler())
solver.run()

print("With RobustScaler")
solver.set_scaler(RobustScaler())
solver.run()



