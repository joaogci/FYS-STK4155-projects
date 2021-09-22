
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel
from RidgeModel import RidgeModel
from LassoModel import LassoModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess
from TrainTestSplitter import TrainTestSplitter

degree = 5

solver = Solver(degree, seed=0)

solver.set_data_generator(PolynomialGenerator(degree=degree, count=100, min_x=-15, max_x=20, noise=0.03))

solver.set_splitter(TrainTestSplitter())

solver.add_model(OLSModel())
solver.add_model(RidgeModel(5))

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
