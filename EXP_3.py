import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
p_disease = 0.01          # P(A)
p_pos_given_dis = 0.95    # P(B|A)
p_pos_given_no_dis = 0.05 # P(B|~A)

p_pos = (p_pos_given_dis * p_disease) + (p_pos_given_no_dis * (1 - p_disease))
p_dis_given_pos = (p_pos_given_dis * p_disease) / p_pos
print(f"Bayes' Theorem P(Disease | Positive): {p_dis_given_pos:.4f}")

# 2. Probability Distribution (Gaussian / Normal)
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, 0, 1)

plt.plot(x, y, color='purple', label='Normal Distribution (μ=0, σ=1)')
plt.title("Normal Probability Distribution")
plt.legend()
plt.show()
