
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error
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

degree = 4

solver = Solver(degree, fit_intercept=True, seed=0)

x_max = 2
x_min = -2
solver.set_data_generator(PolynomialGenerator(degree=degree, count=100, min_x=x_min, normalise=False, max_x=x_max, noise=0.25))

solver.set_scaler(MinMaxScaler())

solver.set_splitter(TrainTestSplitter())

solver.add_model(OLSModel())

solver.add_post_process(ErrDisplayPostProcess())
solver.add_post_process(PlotPostProcess())


solver.run()

solver2 = Solver(degree, fit_intercept=False, seed=0)

x_max = 2
x_min = -2
solver2.set_data_generator(PolynomialGenerator(degree=degree, count=100, min_x=x_min, normalise=False, max_x=x_max, noise=0.25))

solver2.set_scaler(StandardScaler(with_std=False))

solver2.set_splitter(TrainTestSplitter())

solver2.add_model(OLSModel())

solver2.add_post_process(ErrDisplayPostProcess())
solver2.add_post_process(PlotPostProcess())


solver2.run()

# data = solver.get_data()
# x = data[0]
# y = data[1]
# poly = sk.preprocessing.PolynomialFeatures(degree=degree)
# X = poly.fit_transform(x)

# X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y,test_size=0.2,random_state=11)

# scaler = sk.preprocessing.StandardScaler(with_std=False)
# scaler.fit(X_train)
# X_scaled = scaler.transform(X)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled  = scaler.transform(X_test)
# X_train_scaled_own = X_train - np.mean(X_train,axis=0)
# X_test_scaled  = X_test - np.mean(X_train,axis=0)

#scaler = sk.preprocessing.StandardScaler()
# scaler = sk.preprocessing.StandardScaler(with_std=False)
# scaler.fit(y_train)
# print(scaler.mean_)
# print(np.mean(y_train))
# y_scaled = scaler.transform(y)
# y_train_scaled = scaler.transform(y_train)
# y_test_scaled  = scaler.transform(y_test)
# y_train_scaled = y_train - np.mean(y_train)
# y_test_scaled  = y_test - np.mean(y_train)


# linreg_scaled = sk.linear_model.LinearRegression(fit_intercept=False).fit(X_train_scaled, y_train_scaled)
# linreg = sk.linear_model.LinearRegression(fit_intercept=False).fit(X_train, y_train)
# linreg_fit = sk.linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)

# y_tile_scaled = linreg_scaled.predict(X_test_scaled)
# y_tile = linreg.predict(X_test)

# coefs_scale = linreg_scaled.coef_
# inter_scale = linreg_scaled.intercept_
# inter_scale = np.mean(y_train) - np.dot(np.mean(X_train,axis=0),coefs_scale.T)

# coefs = linreg.coef_
# inter = linreg.intercept_

# coefs_fit = linreg_fit.coef_
# inter_fit = linreg_fit.intercept_

# print("Unscaled: ")
# print(coefs)
# print(inter)
# print(f"MSE: {mean_squared_error(y_test, y_tile)}")
# print()
# print("Unscaled (fit_intercept=Ture): ")
# print(coefs_fit)
# print(inter_fit)
# # print(f"MSE: {mean_squared_error(y_test, y_tile)}")
# print()
# print("Scaled: ")
# print(coefs_scale)
# print(inter_scale)
# print(f"MSE: {mean_squared_error(y_test_scaled, y_tile_scaled)}")

# # space = np.linspace(np.min(X_train_scaled[:,1]),np.max(X_train_scaled[:,1]),1000)
# space = np.linspace(np.min(X[:,1]), np.max(X[:,1]),1000)
# out = np.zeros(1000)+inter

# for i, coef in enumerate(coefs[0]):
#     out += coef*space**i

# space_scaled = np.linspace(np.min(X_scaled[:,1]), np.max(X_scaled[:,1]),1000)
# out_scaled = np.zeros(1000)+inter_scale

# for i, coef in enumerate(coefs_scale[0]):
#     out_scaled += coef*space_scaled**i

# plt.figure(1)

# plt.subplot(221)
# plt.title("unscaled")
# plt.plot(space,out)
# plt.plot(X[:, 1], y, 'r*')
# plt.plot(X_test[:, 1], y_tile, 'b*')
# # plt.plot(X_train[:,1],y_train,"r*", alpha=0.25)
# # plt.plot(X_test[:,1],y_test,"r*")

# plt.subplot(222)
# plt.title("scaled")
# plt.plot(space_scaled,out_scaled)
# plt.plot(X_scaled[:, 1], y_scaled, 'r*')
# plt.plot(X_test_scaled[:, 1], y_tile_scaled, 'b*')
# # plt.plot(X_train_scaled[:,1],y_train_scaled,"r*", alpha=0.25)
# # plt.plot(X_test_scaled[:,1],y_test_scaled,"r*")

# plt.subplot(223)
# plt.plot(np.abs(out - out_scaled))

# plt.subplot(224)
# plt.plot(space, out, 'r', label='unscaled')
# plt.plot(space_scaled, out_scaled, 'b', label='scaled')
# plt.legend()

# plt.show()
