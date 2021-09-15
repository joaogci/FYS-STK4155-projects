from Franke import Franke
from Models import Models
import numpy as np
import matplotlib.pyplot as plt


franke = Franke(0, 1, 0.01)

franke.data_set()
franke.plot()
franke.add_noise(dampening_factor=0.1)
# franke.plot()
# plt.show()

lin_model = franke.initialize_regression()

lin_model.design_matrix_2D(5)
lin_model.tt_split(split=0.2)
lin_model.ols_svd()

lin_model.plot_3D()
# franke.plot()
plt.show()
