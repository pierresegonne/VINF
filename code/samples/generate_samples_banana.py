import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import sys

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION
from target_distributions import banana_mu2, banana_std2, banana_std1

SAMPLES_NAME = 'banana'

with pm.Model() as Banana:
    x2 = pm.Normal('x2', mu=banana_mu2, sigma=banana_std2)
    x1 = pm.Normal('x1', mu=(x2 ** 2) / 4, sigma=banana_std1)

with Banana:
    trace = pm.sample(5000, chains=2, tune=3000, target_accept=0.9)

print(pm.summary(trace).round(2))
pm.traceplot(trace)

plt.figure()
plt.scatter(trace['x1'], trace['x2'], color='darkorchid', alpha=0.6)
plt.xlabel(r'$z_{1}$')
plt.ylabel(r'$z_{2}$')

plt.figure()
plt.hexbin(trace['x1'], trace['x2'], gridsize=50, mincnt=1, bins='log', cmap='inferno')
plt.colorbar()
plt.xlabel(r'$z_{1}$')
plt.ylabel(r'$z_{2}$')
plt.show()

with open(f"{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'wb') as f:
    pos = np.hstack((trace['x1'].reshape(-1, 1), trace['x2'].reshape(-1, 1)))
    np.save(f, pos.astype(np.float32))
    f.close()

plt.show()
