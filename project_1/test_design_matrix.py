from Regression import Regression
import numpy as np

x = np.array([1,2,3,4])
reg = Regression()

design_mat = reg.design_matrix(x,2)
true_mat = np.array([[1,1,1],
                     [1,2,4],
                     [1,3,9],
                     [1,4,16]])

for i in range(4):
    for j in range(2+1):
        if design_mat[i,j] != true_mat[i,j]:
            print("NO!!!")

x = np.array([1,2,3])
y = np.array([4,5,6])
x,y = np.meshgrid(x,y)

design_mat_2D = reg.design_matrix_2D(x,y,2)
true_mat_2D = np.array([[1,1,4,1,4, 16],
                        [1,2,4,4,8, 16],
                        [1,3,4,9,12,16],
                        [1,1,5,1,5, 25],
                        [1,2,5,4,10,25],
                        [1,3,5,9,15,25],
                        [1,1,6,1,6, 36],
                        [1,2,6,4,12,36],
                        [1,3,6,9,18,36]])

for i in range(9):
    for j in range(5+1):
        if design_mat_2D[i,j] != true_mat_2D[i,j]:
            print("NO!!!")

"""
dm = [[1,1,1]
      [1,2,4]
      [1,3,9]
      [1,4,16]]


after meshgrid
x = [[1 2 3]
     [1 2 3]
     [1 2 3]]

x = [1 2 3 1 2 3 1 2 3]

y = [[4 4 4]
    [5 5 5]
    [6 6 6]]

y = [4 4 4 5 5 5 6 6 6]

(1,x,y,x**2,xy,y**2)

dm = [[1,1,4,1,4, 16]
     [1,2,4,4,8, 16]
     [1,3,4,9,12,16]
     [1,1,5,1,5, 25]
     [1,2,5,4,10,25]
     [1,3,5,9,15,25]
     [1,1,6,1,6, 36]
     [1,2,6,4,12,36]
     [1,3,6,9,18,36]]
"""