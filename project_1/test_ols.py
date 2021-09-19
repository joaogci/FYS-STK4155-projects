
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel
from OLSSVDModel import OLSSVDModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess
from TrainTestSplitter import TrainTestSplitter


solver = Solver(3)

solver.set_data_generator(PolynomialGenerator(degree=3, count=100, min_x=-5, max_x=5, noise=0.05))

solver.set_splitter(TrainTestSplitter())

solver.set_model(OLSSVDModel())

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
