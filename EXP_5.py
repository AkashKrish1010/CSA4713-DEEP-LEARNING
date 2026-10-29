import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.linspace(0, 1, 30)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
x_plot = np.linspace(0, 1, 100)

# Models: Underfitting (deg 1), Good Fit (deg 3), Overfitting (deg 15)
degrees = [1, 3, 15]
titles = ['Underfitting (High Bias)', 'Good Fit (Balanced)', 'Overfitting (High Variance)']

plt.figure(figsize=(12, 4))
for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y)
    plt.subplot(1, 3, i + 1)
    plt.scatter(x, y, color='black', s=15, label='Data')
    plt.plot(x_plot, model.predict(x_plot[:, None]), color='red', label=f'Degree {d}')
    plt.title(titles[i])
    plt.legend()

plt.tight_layout()
plt.show()
