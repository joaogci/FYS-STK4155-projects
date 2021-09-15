from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 100
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

linreg = Models()
X = linreg.design_matrix(x, degree)
pred, beta = linreg.ols(X, y, use_svd=False, pseudo_inverse=True, verbose=False)
pred_svd, _ = linreg.ols(X, y, use_svd=True, verbose=False)

# Show data & prediction
plt.plot(x, y ,'k+', label='Input data')
plt.plot(np.sort(x, 0), np.sort(pred, 0), 'b-', label='Prediction from matrix inverse OLS')
plt.plot(np.sort(x, 0), np.sort(pred_svd, 0), 'r--', label='Prediction from SVD OLS')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ordinary Least Squares predictions')
plt.legend()
plt.show()
