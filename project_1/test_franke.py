
from Solver import Solver
from FrankeGenerator import FrankeGenerator
from TrainTestSplitter import TrainTestSplitter
from OLSModel import OLSModel
from RidgeModel import RidgeModel
from LassoModel import LassoModel
from ErrDisplayPostProcess import ErrDisplayPostProcess
from PlotPostProcess import PlotPostProcess


solver = Solver(5)

solver.set_data_generator(FrankeGenerator(0, 1, 20, random=True, noise=0.01))

solver.set_splitter(TrainTestSplitter())

solver.add_model(OLSModel())
solver.add_model(RidgeModel(0.001))
solver.add_model(LassoModel(0.001))

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
