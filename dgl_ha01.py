
import numpy as np
import matplotlib.pyplot as plt

L =  np.array([[1, 0, 0],
               [-2, 1, 0],
               [4, 5, 1]])

R = np.array([[2, -1, 6],
              [0, 3, 9],
              [0, 0, -2]])

b = np.array([[18],
              [-3],
              [231]])


n = L.shape[0]
y = np.zeros((n, 1))

for i in range(n):
    y[i] = b[i]

    for j in range(i):
        y[i] = y[i] - L[i, j] * y[j]

    y[i] = y[i] / L[i, i]

print(y)
