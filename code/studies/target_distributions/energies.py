import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from scipy.special import expit


def w1(z):
    z1 = z[:, 0]
    return np.sin(2 * np.pi * z1 / 4)


def w2(z):
    z1 = z[:, 0]
    exp_arg = -0.5 * ((z1 - 1) / 0.6) ** 2
    return 3 * np.exp(exp_arg)


def w3(z):
    z1 = z[:, 0]
    return 3 * expit((z1 - 1) / 0.3)


def energy_1_log_pdf(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = (z1 ** 2 + z2 ** 2) ** 0.5
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    u = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)

    return -u


def pot1f(z):
    return np.exp(-energy_1_log_pdf(z))


def energy_2_log_pdf(z):
    z2 = z[:, 1]
    return - 0.5 * ((z2 - w1(z)) / 0.4) ** 2


def pot2f(z):
    return np.exp(-energy_2_log_pdf(z))


def energy_3_log_pdf(z):
    z2 = z[:, 1]

    x1 = -0.5 * ((z2 - w1(z)) / 0.35) ** 2
    x2 = -0.5 * ((z2 - w1(z) + w2(z)) / 0.35) ** 2
    a = np.maximum(x1, x2)
    exp1 = np.exp(x1 - a)
    exp2 = np.exp(x2 - a)
    return a + np.log(exp1 + exp2)


def pot3f(z):
    return np.exp(-energy_3_log_pdf(z))


def energy_4_log_pdf(z):
    z2 = z[:, 1]
    x1 = -0.5 * ((z2 - w1(z)) / 0.4) ** 2
    x2 = -0.5 * ((z2 - w1(z) + w3(z)) / 0.35) ** 2
    a = np.maximum(x1, x2)
    exp1 = np.exp(x1 - a)
    exp2 = np.exp(x2 - a)
    return a + np.log(exp1 + exp2)  # Try adding a small value to prevent


def pot4f(z):
    return np.exp(-energy_4_log_pdf(z))


def contour_pot(potf, ax=None, title=None, xlim=5, ylim=5):
    grid = pm.floatX(np.mgrid[-xlim:xlim:100j, -ylim:ylim:100j])
    grid_2d = grid.reshape(2, -1).T
    cmap = plt.get_cmap('inferno')
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 9))
    pdf1e = np.exp(-potf(grid_2d))
    contour = ax.contourf(grid[0], grid[1], pdf1e.reshape(100, 100), cmap=cmap)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return ax


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
contour_pot(pot1f, ax[0], 'Energy 1')
contour_pot(pot2f, ax[1], 'Energy 2')
contour_pot(pot3f, ax[2], 'Energy 3')
contour_pot(pot4f, ax[3], 'Energy 4')
fig.tight_layout()

plt.show()
