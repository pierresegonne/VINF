import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION
from target_distributions import eight_schools_y, eight_schools_sigma

SAMPLES_NAME = 'eight_schools'

N = eight_schools_y.shape[0]

with pm.Model() as NonCentered_eight:
    tau = pm.HalfCauchy('tau', beta=5)
    mu = pm.Normal('mu', mu=0, sigma=5)
    theta_tilde = pm.Normal('theta_t', mu=0, sigma=1, shape=N)
    theta = pm.Deterministic('theta', mu + tau * theta_tilde)
    y = pm.Normal('obs', mu=theta, sigma=eight_schools_sigma, observed=eight_schools_y)

with NonCentered_eight:
    noncentered_trace = pm.sample(5000, chains=2, tune=1000, target_accept=.80)

print('Non Centered')
print(pm.summary(noncentered_trace).round(2))
pm.traceplot(noncentered_trace)

plt.figure()
plt.scatter(np.log(noncentered_trace['tau'][1000:]), noncentered_trace['theta'][:,0][1000:], color='green')
plt.legend(['Non-Centered'])
plt.show()

with open(f"{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'wb') as f:
    pickle.dump(noncentered_trace, f)
    f.close()
