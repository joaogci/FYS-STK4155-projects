
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel
from OLSSVDModel import OLSSVDModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess
from TrainTestSplitter import TrainTestSplitter


solver = Solver(5, seed=0)

solver.set_data_generator(PolynomialGenerator(degree=5, count=100, min_x=-15, max_x=20, noise=0.03))

solver.set_splitter(TrainTestSplitter())

solver.set_model(OLSSVDModel())

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
