from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 100
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

linreg = Models(x, y, verbose=False)
linreg.design_matrix(degree)
linreg.ols(pseudo_inverse=True)
linreg.print_error_estimates('regular OLS')

linregSVD = Models(x, y, verbose=False)
linregSVD.design_matrix(degree)
linregSVD.ols_svd()
linregSVD.print_error_estimates('SVD')

# Show data & prediction
plt.plot(x, y ,'k+', label='Input data')
plt.plot(np.sort(x, 0), np.sort(linreg.prediction, 0), 'b-', label='Prediction from matrix inverse OLS')
plt.plot(np.sort(x, 0), np.sort(linregSVD.prediction, 0), 'r--', label='Prediction from SVD OLS')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ordinary Least Squares predictions')
plt.legend()
plt.show()
