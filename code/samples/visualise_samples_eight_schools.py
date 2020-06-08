import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from scipy.stats import halfcauchy, norm

SAMPLES_NAME = 'eight_schools.npy'
Y = np.array([28, 8, -3, 7, -1, 1, 18, 12])

with open(SAMPLES_NAME, 'rb') as f:
    mcmc_trace = pickle.load(f)

# THETA - LOG TAU
fig, axs = plt.subplots(2, 2)

axs[0, 0].scatter(np.log(mcmc_trace['tau'][1000:]), mcmc_trace['theta'][:, 0][1000:], color='lightsteelblue', alpha=0.6)
axs[0, 0].axhline(y=Y[0], color='black', linestyle='-')
axs[0, 0].set_xlabel(r'log($\tau$)')
axs[0, 0].set_ylabel(r'$\theta_{1}$')
axs[0, 0].annotate(r'$y_{1}$', xy=(-4, Y[0] + 1), xycoords='data')

axs[0, 1].scatter(np.log(mcmc_trace['tau'][1000:]), mcmc_trace['theta'][:, 1][1000:], color='lightsteelblue', alpha=0.6)
axs[0, 1].axhline(y=Y[1], color='black', linestyle='-')
axs[0, 1].set_xlabel(r'log($\tau$)')
axs[0, 1].set_ylabel(r'$\theta_{2}$')
axs[0, 1].annotate(r'$y_{2}$', xy=(-4, Y[1] + 1), xycoords='data')

axs[1, 0].scatter(np.log(mcmc_trace['tau'][1000:]), mcmc_trace['theta'][:, 2][1000:], color='lightsteelblue', alpha=0.6)
axs[1, 0].axhline(y=Y[2], color='black', linestyle='-')
axs[1, 0].set_xlabel(r'log($\tau$)')
axs[1, 0].set_ylabel(r'$\theta_{3}$')
axs[1, 0].annotate(r'$y_{3}$', xy=(-4, Y[2] + 1), xycoords='data')

axs[1, 1].scatter(np.log(mcmc_trace['tau'][1000:]), mcmc_trace['theta'][:, -2][1000:], color='lightsteelblue',
                  alpha=0.6)
axs[1, 1].axhline(y=Y[-2], color='black', linestyle='-')
axs[1, 1].set_xlabel(r'log($\tau$)')
axs[1, 1].set_ylabel(r'$\theta_{7}$')
axs[1, 1].annotate(r'$y_{7}$', xy=(-4, Y[-2] + 1), xycoords='data')

# THETA - MU
fig, axs = plt.subplots(2, 2)

axs[0, 0].scatter(np.log(mcmc_trace['mu'][1000:]), mcmc_trace['theta'][:, 0][1000:], color='tan', alpha=0.6)
axs[0, 0].axhline(y=Y[0], color='black', linestyle='-')
axs[0, 0].set_xlabel(r'$\mu$')
axs[0, 0].set_ylabel(r'$\theta_{1}$')
axs[0, 0].annotate(r'$y_{1}$', xy=(-4, Y[0] + 1), xycoords='data')

axs[0, 1].scatter(np.log(mcmc_trace['mu'][1000:]), mcmc_trace['theta'][:, 1][1000:], color='tan', alpha=0.6)
axs[0, 1].axhline(y=Y[1], color='black', linestyle='-')
axs[0, 1].set_xlabel(r'$\mu$')
axs[0, 1].set_ylabel(r'$\theta_{2}$')
axs[0, 1].annotate(r'$y_{2}$', xy=(-4, Y[1] + 1), xycoords='data')

axs[1, 0].scatter(np.log(mcmc_trace['mu'][1000:]), mcmc_trace['theta'][:, 2][1000:], color='tan', alpha=0.6)
axs[1, 0].axhline(y=Y[2], color='black', linestyle='-')
axs[1, 0].set_xlabel(r'$\mu$')
axs[1, 0].set_ylabel(r'$\theta_{3}$')
axs[1, 0].annotate(r'$y_{3}$', xy=(-4, Y[2] + 1), xycoords='data')

axs[1, 1].scatter(np.log(mcmc_trace['mu'][1000:]), mcmc_trace['theta'][:, -2][1000:], color='tan',
                  alpha=0.6)
axs[1, 1].axhline(y=Y[-2], color='black', linestyle='-')
axs[1, 1].set_xlabel(r'$\mu$')
axs[1, 1].set_ylabel(r'$\theta_{7}$')
axs[1, 1].annotate(r'$y_{7}$', xy=(-4, Y[-2] + 1), xycoords='data')

# MU & THETA
mu_true_pdf = norm(loc=0, scale=5).pdf
mu_linspace = np.linspace(-15, 15, 2000)
tau_true_pdf = halfcauchy(loc=0, scale=5).pdf
tau_linspace = np.linspace(-2, 30, 2000)

f, (ax1, ax2) = plt.subplots(1, 2)
sns.kdeplot(mcmc_trace['mu'][1000:], color='steelblue', ax=ax1, label=r'kde($\mu$)')
count, bins, ignored = ax1.hist(mcmc_trace['mu'][1000:], 50, density=True, color='skyblue', alpha=0.6)
ax1.plot(mu_linspace, mu_true_pdf(mu_linspace), color='firebrick', label=r'$\mathcal{N}(0,5)$')
ax1.title.set_text(r'Distribution $\mu$')
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'p($\mu$)')
ax1.legend()

sns.kdeplot(mcmc_trace['tau'][1000:], color='steelblue', ax=ax2, label=r'kde($\tau$)')
count, bins, ignored = ax2.hist(mcmc_trace['tau'][1000:], 50, density=True, color='skyblue', alpha=0.6)
ax2.plot(tau_linspace, tau_true_pdf(tau_linspace), color='firebrick', label=r'Half-Cauchy$(0,5)$')
ax2.title.set_text(r'Distribution $\tau$')
ax2.set_xlabel(r'$\tau$')
ax2.set_ylabel(r'p($\tau$)')
ax2.legend()

plt.show()
