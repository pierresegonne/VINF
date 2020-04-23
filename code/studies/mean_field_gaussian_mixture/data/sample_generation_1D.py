import matplotlib.pyplot as plt
import numpy as np

theta = np.random.normal(loc=0, scale=1, size=1000)
eps = np.random.normal(loc=0, scale=0.1, size=1000)
y = theta**2 + eps

plt.scatter([i for i in range(len(y))], y)
plt.show()