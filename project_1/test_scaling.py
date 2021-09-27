
from include.Solver import Solver
from include.PolynomialGenerator import PolynomialGenerator
from include.OLSModel import OLSModel
from include.RidgeModel import RidgeModel
from include.LassoModel import LassoModel
from include.ErrDisplayPostProcess import ErrDisplayPostProcess
from include.PlotPostProcess import PlotPostProcess
from include.TrainTestSplitter import TrainTestSplitter
from include.StandardScaler import StandardScaler
from include.ExponentialGenerator import ExponentialGenerator
from include.MinMaxScaler import MinMaxScaler
from include.RobustScaler import RobustScaler

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



