import numpy as np
import tensorflow as tf

from numpy import pi
from tensorflow import math
from tensorflow_probability import distributions as tfd

AVAILABLE_1D_DISTRIBUTIONS = ['', 'two_hills']
AVAILABLE_2D_DISTRIBUTIONS = ['', 'banana', 'circle', 'eight_schools', 'figure_eight']

def pdf_2D(z, density_name=''):
    assert density_name in AVAILABLE_2D_DISTRIBUTIONS, "Incorrect density name."
    if density_name == '':
        return 1
    elif density_name == 'banana':
        z1, z2 = z[:, 0], z[:, 1]
        mu = np.array([0.5,0.5], dtype='float32')
        cov = np.array([[0.06,0.055],[0.055,0.06]], dtype='float32')
        scale = tf.linalg.cholesky(cov)
        p = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale)
        z2 = z1**2 + z2
        z1, z2 = tf.expand_dims(z1, 1), tf.expand_dims(z2, 1)
        z = tf.concat([z1, z2], axis=1)
        return p.prob(z)
    elif density_name == 'circle':
        z1, z2 = z[:, 0], z[:, 1]
        norm = (z1**2 + z2**2)**0.5
        exp1 = math.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
        exp2 = math.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
        u = 0.5 * ((norm - 4) / 0.4) ** 2 - math.log(exp1 + exp2)
        return math.exp(-u)
    elif density_name == 'eight_schools':
        y_i = 10
        sigma_i = 10
        thetas, mu, tau  = z[:, 0], z[:, 1], tf.maximum(z[:, 2], 0 + 1e-10)
        likelihood = tfd.Normal(loc=thetas, scale=sigma_i)
        prior_theta = tfd.Normal(loc=mu, scale=tau)
        prior_mu = tfd.Normal(loc=0, scale=5)
        prior_tau = tfd.HalfCauchy(loc=0, scale=5)
        return likelihood.prob(y_i) * prior_theta.prob(thetas) * prior_mu.prob(mu) * prior_tau.prob(tau)
    elif density_name == 'figure_eight':
        mu1 = 1 * np.array([-1,-1], dtype='float32')
        mu2 = 1 * np.array([1,1], dtype='float32')
        scale = 0.45 * np.array([1,1], dtype='float32')
        pi = 0.5
        comp1 = tfd.MultivariateNormalDiag(loc=mu1, scale_diag=scale)
        comp2 = tfd.MultivariateNormalDiag(loc=mu2, scale_diag=scale)
        return (1-pi)*comp1.prob(z) + pi*comp2.prob(z)

def pdf_1D(z, density_name=''):
    assert density_name in AVAILABLE_1D_DISTRIBUTIONS, "Incorrect density name."
    if density_name == '':
        return 1
    elif density_name == 'two_hills':
        y = 0.5
        sigma2 = 0.1
        likelihood = (1/math.sqrt(2*pi*sigma2))*math.exp(-((y-(z**2))**2)/(2*sigma2))
        prior = (1/math.sqrt(2*pi))**math.exp(-(z**2)/2)
        return likelihood*prior

