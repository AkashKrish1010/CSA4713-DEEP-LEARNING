import numpy as np
import matplotlib.pyplot as plt

# 1. Vector Operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("Vector Dot Product:", np.dot(v1, v2))

# 2. Matrix Operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix Multiplication:\n", np.matmul(A, B))
print("Transpose of A:\n", A.T)
print("Element-wise Multiplication:\n", A * B)

# 3. Visualization
plt.imshow(A, cmap='Blues')
plt.title("Matrix A Heatmap")
plt.colorbar()
plt.show()
