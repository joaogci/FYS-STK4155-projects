import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

n = 100

np.random.seed(10)
x = np.linspace(-4, 4, n).reshape(-1, 1)
y = x ** 2 * np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + 0.5 * np.random.normal(0, 1, x.shape)

poly = PolynomialFeatures(degree=10)
X = poly.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = StandardScaler()
scaler.fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)


X_train_scaled[:, 0] = 1
X_test_scaled[:, 0] = 1

lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train_scaled, y_train)
y_pred_train = lin_reg.predict(X_train_scaled)
y_pred_test = lin_reg.predict(X_test_scaled)

print("scaled train: ", mean_squared_error(y_train_scaled, y_pred_train))
print("scaled test : ", mean_squared_error(y_test_scaled, y_pred_test))

betas_scaled = lin_reg.coef_

lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train, y_train)
y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

print("unscaled train: ", mean_squared_error(y_train, y_pred_train))
print("unscaled test : ", mean_squared_error(y_test, y_pred_test))

betas = lin_reg.coef_

print(betas)
print(betas_scaled)

plt.figure(1)

plt.plot(X_train[:,1], y_train, 'k+', label='Input data (training set)', alpha=0.25)
plt.plot(X_test[:,1], y_test, 'k+', label='Input data (test set)')

x_display = np.linspace(-4, 4, 10000)
y_display = np.zeros(10000)
for i in range(len(betas[0])):
    y_display += betas[0][i] * x_display ** i
plt.plot(x_display, y_display, '--', label='Prediction (unscaled)')

plt.legend()

plt.figure(2)

plt.plot(X_train_scaled[:,1], y_train_scaled, 'k+', label='Input data (training set)', alpha=0.25)
plt.plot(X_test_scaled[:,1], y_test_scaled, 'k+', label='Input data (test set)')

x_display = np.linspace(-1, 1, 10000)
y_display = np.zeros(10000)
for i in range(len(betas_scaled[0])):
    y_display += betas_scaled[0][i] * x_display ** i
plt.plot(x_display, y_display, '--', label='Prediction (scaled)')

plt.legend()

plt.show()
