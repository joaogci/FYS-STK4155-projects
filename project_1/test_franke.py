from Franke import Franke
import matplotlib.pyplot as plt

franke = Franke(-2, 2, 0.1, random=True)
franke.data_set()
franke.plot()
plt.show()
