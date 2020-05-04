import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

y_observed = np.array([28,8,-3,7,-1,1,18,12])
sigma_observed = np.array([15,10,16,11,9,11,10,18])
N = y_observed.shape[0]

with pm.Model() as Centered_eight:
    tau = pm.HalfCauchy('tau', beta=5)
    mu = pm.Normal('mu', mu=0, sigma=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=N)
    y = pm.Normal('y', mu=theta, sigma=sigma_observed, observed=y_observed)

with pm.Model() as NonCentered_eight:
    tau = pm.HalfCauchy('tau', beta=5)
    mu = pm.Normal('mu', mu=0, sigma=5)
    theta_tilde = pm.Normal('theta_t', mu=0, sigma=1, shape=N)
    theta = pm.Deterministic('theta', mu + tau * theta_tilde)
    y = pm.Normal('obs', mu=theta, sigma=sigma_observed, observed=y_observed)

with Centered_eight:
    centered_trace = pm.sample(5000, chains=2, tune=2000, target_accept=.99)

with NonCentered_eight:
    noncentered_trace = pm.sample(5000, chains=2, tune=1000, target_accept=.80)

print('Centered')
print(pm.summary(centered_trace).round(2))
pm.traceplot(centered_trace)

print('Non Centered')
print(pm.summary(noncentered_trace).round(2))
pm.traceplot(noncentered_trace)


plt.figure()
plt.scatter(np.log(centered_trace['tau'][2000:]), centered_trace['theta'][:,0][2000:], color='orange')
plt.scatter(np.log(noncentered_trace['tau'][2000:]), noncentered_trace['theta'][:,0][2000:], color='green')
plt.legend(['Centered', 'Non-Centered'])

plt.show()
