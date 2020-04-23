import numpy as np
import tensorflow as tf

from numpy import pi
from tensorflow import math
from tensorflow_probability import distributions as tfd

# TODO transform all pdf to log_pdf
# TODO handle posterior parameters better
def two_hills_log_pdf(z):
    y = 0.5
    sigma2 = 0.1

    # likelihood = (1/math.sqrt(2*pi*sigma2))*math.exp(-((y-(z**2))**2)/(2*sigma2))
    likelihood = tfd.Normal(loc=z**2, scale=math.sqrt(sigma2))
    # prior = (1/math.sqrt(2*pi))**math.exp(-(z**2)/2)
    prior = tfd.Normal(loc=0, scale=1)

    return likelihood.log_prob(y) + prior.log_prob(z)

def banana_log_pdf(z):
    mu = np.array([0.5,0.5], dtype='float32')
    cov = np.array([[0.06,0.055],[0.055,0.06]], dtype='float32')
    scale = tf.linalg.cholesky(cov)

    z1, z2 = z[:, 0], z[:, 1]

    p = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale)
    z2 = z1**2 + z2
    z1, z2 = tf.expand_dims(z1, 1), tf.expand_dims(z2, 1)
    z = tf.concat([z1, z2], axis=1)

    return p.log_prob(z)

def circle_log_pdf(z):
    z1, z2 = z[:, 0], z[:, 1]

    norm = (z1**2 + z2**2)**0.5
    exp1 = math.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
    exp2 = math.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - math.log(exp1 + exp2)

    return -u

def figure_eight_log_pdf(z):
    mu1 = 1 * np.array([-1,-1], dtype='float32')
    mu2 = 1 * np.array([1,1], dtype='float32')
    scale = 0.45 * np.array([1,1], dtype='float32')
    pi = 0.5

    comp1 = tfd.MultivariateNormalDiag(loc=mu1, scale_diag=scale)
    comp2 = tfd.MultivariateNormalDiag(loc=mu2, scale_diag=scale)

    return math.log((1-pi)*comp1.prob(z) + pi*comp2.prob(z))

def eight_schools_log_pdf(z):
    y_i = 0
    sigma_i = 10

    thetas, mu, log_tau  = z[:, 0], z[:, 1], z[:, 2]

    likelihood = tfd.Normal(loc=thetas, scale=sigma_i)
    prior_theta = tfd.Normal(loc=mu, scale=math.exp(log_tau))
    prior_mu = tfd.Normal(loc=0, scale=5)
    prior_tau = tfd.HalfCauchy(loc=0, scale=5)
    det_jac = math.log(math.exp(log_tau)) # kept log(exp()) for mathematical understanding.

    return likelihood.log_prob(y_i) + prior_theta.log_prob(thetas) + prior_mu.log_prob(mu) + prior_tau.log_prob(math.exp(log_tau)) + log_det_jac


def get_log_joint_pdf(target_distribution):

    if target_distribution == 'two_hills':
        log_joint_pdf = two_hills_log_pdf
    elif target_distribution == 'banana':
        log_joint_pdf = banana_log_pdf
    elif target_distribution == 'circle':
        log_joint_pdf = circle_log_pdf
    elif target_distribution == 'figure_eight':
        log_joint_pdf = figure_eight_log_pdf
    elif target_distribution == 'eight_schools':
        log_joint_pdf = eight_schools_log_pdf

    return log_joint_pdf

