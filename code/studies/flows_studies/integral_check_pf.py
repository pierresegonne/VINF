import matplotlib.pyplot as plt
import numpy as np

# -20, 20, -20, 20, 1000x1000
# banana
# First column = nbr flows, others are resulting integrals
integrals = np.array([
    [1, .65, .97, 1.1],
    [2, .6, .24, .45],
    [4, .34, .43, .78],
    [8, .92, .35, 1.83]
])

for i in range(integrals.shape[0]):
    print(f"n_flows {integrals[i, 0]}, average integral value {np.mean(integrals[i, 1:])}")


# -5, 25, -10, 10, 1000x1000
# banana
# Check for correlation between variance of q0 and resulting integral
scales = np.array([
    [1.0578514, 1.1445957],
    [1.3237904, 0.7846588],
    [1.3476086,  0.83441836],
    [1.6165129,  0.51923436],
    [1.8854616, 1.4038025],
    [1.8393,    1.9363197],
    [1.3402703, 1.6748267],
    [1.0575297,  0.75540084],
])
integrals = np.array([
    [.48],
    [.41],
    [.42],
    [.32],
    [1.03],
    [1.58],
    [.98],
    [.31],
])

plt.figure()
plt.scatter(integrals, scales[:, 0] * scales[:, 1])
plt.show()