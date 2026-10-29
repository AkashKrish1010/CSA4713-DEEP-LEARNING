import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Distributions P (True) and Q (Approximation)
P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.6, 0.3, 0.1])

# Calculations
h_p = entropy(P)             # Shannon Entropy
kl_div = entropy(P, Q)       # KL Divergence
cross_entropy = h_p + kl_div # Cross-Entropy

print(f"Entropy H(P): {h_p:.4f}")
print(f"KL Divergence D_KL(P || Q): {kl_div:.4f}")
print(f"Cross-Entropy H(P, Q): {cross_entropy:.4f}")

# Visualization
classes = ['Class 1', 'Class 2', 'Class 3']
x = np.arange(len(classes))
plt.bar(x - 0.15, P, width=0.3, label='P (True)')
plt.bar(x + 0.15, Q, width=0.3, label='Q (Predicted)')
plt.xticks(x, classes)
plt.title("Distributions P vs Q")
plt.legend()
plt.show()
