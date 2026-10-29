import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Sample Data using Pandas
df = pd.DataFrame({
    'X': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'Y': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
})

# 2. Covariance Matrix
cov_mat = df.cov()
print("Covariance Matrix:\n", cov_mat)

# 3. Eigenvalues and Eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print("Eigenvalues:\n", eig_vals)
print("Eigenvectors:\n", eig_vecs)

# 4. Visualization
plt.scatter(df['X'], df['Y'], color='blue', label='Data')
origin = [df['X'].mean(), df['Y'].mean()]
plt.quiver(*origin, *eig_vecs[:, 0], color='red', scale=3, label='Eigenvector 1')
plt.quiver(*origin, *eig_vecs[:, 1], color='green', scale=3, label='Eigenvector 2')
plt.title("Covariance, Eigenvalues & Eigenvectors")
plt.legend()
plt.show()
