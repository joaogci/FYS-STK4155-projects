
from Solver import Solver
from PolynomialGenerator import PolynomialGenerator
from OLSModel import OLSModel
from OLSSVDModel import OLSSVDModel

solver = Solver(3)
solver.set_data_generator(PolynomialGenerator(degree=3, count=100, min_x=-5, max_x=5, noise=10))
solver.set_model(OLSSVDModel())
solver.run()
