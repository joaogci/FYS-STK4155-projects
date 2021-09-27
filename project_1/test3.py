import numpy as np

x = np.arange(0, 10)
y = np.arange(4, 14)
print(x.shape, y.shape)
K = 5
train_KFold_size = int(x.shape[0] / K)

for k in range(K):
    print( x[np.concatenate((np.arange(0, k * train_KFold_size), np.arange((k+1) * train_KFold_size, K * train_KFold_size)))] )
    
    print( x[np.arange(k * train_KFold_size, (k+1) * train_KFold_size)] )

# Output dictionaries
x_KFold = dict()
y_KFold = dict()

# Split into k folds
KFold_size = np.floor(y.shape[0] / K)

for k in range(K):
    train_idx = np.concatenate((np.arange(0, k * KFold_size, dtype=int), np.arange((k+1) * KFold_size, K * KFold_size, dtype=int)))
    test_idx = np.arange(k * KFold_size, (k+1) * KFold_size, dtype=int)

    x_KFold['train_' + str(k)] = x[train_idx]
    x_KFold['test_' + str(k)] = x[test_idx]
    
    y_KFold['train_' + str(k)] = y[train_idx]
    y_KFold['test_' + str(k)] = y[test_idx]

print(x_KFold)
print(y_KFold)

