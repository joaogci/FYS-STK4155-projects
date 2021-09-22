
from Solver import Solver
from FrankeGenerator import FrankeGenerator
from TrainTestSplitter import TrainTestSplitter
from OLSModel import OLSModel
from RidgeModel import RidgeModel
from LassoModel import LassoModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess


solver = Solver(5)

solver.set_data_generator(FrankeGenerator(0, 1, 0.025, random=True, noise=0.01))

solver.set_splitter(TrainTestSplitter())

solver.set_model(RidgeModel(0.001))

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
