
from include.Solver import Solver
from include.FrankeGenerator import FrankeGenerator
from include.TrainTestSplitter import TrainTestSplitter
from include.OLSModel import OLSModel
from include.RidgeModel import RidgeModel
from include.LassoModel import LassoModel
from include.ErrDisplayPostProcess import ErrDisplayPostProcess
from include.PlotPostProcess import PlotPostProcess


solver = Solver(5)

solver.set_data_generator(FrankeGenerator(0, 1, 20, random=True, noise=0.01))

solver.set_splitter(TrainTestSplitter())

for i in range(1):
    solver.add_model(RidgeModel(0.01 * i))

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())

solver.run()
