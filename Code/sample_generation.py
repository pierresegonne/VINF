"""
Scripts to generate data samples.
"""
import numpy as np
import matplotlib.pyplot as plt

from serialization import serialize_object

def generate_spsd(n):
    m = np.random.uniform(0,2,size=n)
    m = m@(m.T)
    m = m + n*np.eye(n)
    return m

def multivariate_gaussian_2d(n_components, n_samples=100, span=[-10,10]):
    mus = []
    sigmas = []
    X = np.zeros(((n_samples*n_components),2))
    for n in range(n_components):
        mus.append(np.random.uniform(span[0],span[1],size=(1,2))[0])
        sigmas.append(generate_spsd(2))
        X[n*n_samples:(n+1)*n_samples,:] = np.random.multivariate_normal(mus[-1],sigmas[-1], size=n_samples)
    return X, mus, sigmas



X, mus, sigmas = multivariate_gaussian_2d(2, n_samples=300)
SAVE_DATA_NAME = 'data/mvg2D_X.pickle'
SAVE_PARAMS_NAME = 'data/mvg2D_params.pickle'
serialize_object(X, SAVE_DATA_NAME)
serialize_object({'mus': mus, 'sigmas': sigmas}, SAVE_PARAMS_NAME)

# Visualization to check interest of samples.
plt.plot(X[:,0], X[:,1], 'o')
plt.show()


