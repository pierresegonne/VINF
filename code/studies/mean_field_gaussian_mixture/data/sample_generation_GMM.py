"""
Scripts to generate data samples.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from scipy.stats import multivariate_normal
from serialisation import serialise_object

def generate_spsd(n):
    m = np.random.uniform(0,3,size=n)
    m = m@(m.T)
    m = m + n*np.eye(n)
    if np.random.uniform(0,1) > .5:
        m[0,1] = -m[0,1]
        m[1, 0] = -m[1, 0]
    return m

def multivariate_gaussian_2d(n_components, n_samples=100, span=[-10,10]):
    mus = []
    sigmas = []
    X = np.zeros(((n_samples*n_components),2))

    # Create grid for contours
    x_c = np.linspace(2*span[0],2*span[1],100)
    y_c = np.linspace(2*span[0],2*span[1],100)
    X_c, Y_c = np.meshgrid(x_c,y_c)
    pos = np.empty(X_c.shape + (2,))
    pos[:, :, 0] = X_c; pos[:, :, 1] = Y_c
    contours = {'X': X_c, 'Y': Y_c, 'c': []}

    for n in range(n_components):
        mus.append(np.random.uniform(span[0],span[1],size=(1,2))[0])
        sigmas.append(generate_spsd(2))
        X[n*n_samples:(n+1)*n_samples,:] = np.random.multivariate_normal(mus[-1],sigmas[-1], size=n_samples)
        contours['c'].append(multivariate_normal(mus[-1],sigmas[-1]).pdf(pos))
    return X, mus, sigmas, contours



X, mus, sigmas, contours = multivariate_gaussian_2d(2, n_samples=200)
SAVE_DATA_NAME = '2D_gaussian_mixture_X.pickle'
SAVE_PARAMS_NAME = '2D_gaussian_mixture_params.pickle'
SAVE_CONTOURS_NAME = '2D_gaussian_mixture_contours.pickle'
serialise_object(X, SAVE_DATA_NAME)
serialise_object({'mus': mus, 'sigmas': sigmas}, SAVE_PARAMS_NAME)
serialise_object(contours, SAVE_CONTOURS_NAME)

# Visualization to check interest of samples.
plt.plot(X[:,0], X[:,1], 'o', color='grey', alpha=0.5)
for contour in contours['c']:
    plt.contour(contours['X'], contours['Y'], contour, levels=6, cmap='magma')
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.show()


