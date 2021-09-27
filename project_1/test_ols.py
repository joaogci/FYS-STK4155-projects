
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

solver = Solver(3, fit_intercept=True)
solver.set_data_generator(PolynomialGenerator(count=100, min_x=-15, max_x=15, noise=0.01))
solver.add_model(OLSModel(0.01))
plotter = PlotPostProcess(show=False, title='fit_intercept=True')
solver.add_post_process(plotter)
solver.run()

solver.fit_intercept = False
plotter.show=True
plotter.title='fit_intercept=False'
solver.run()
