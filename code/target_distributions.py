import numpy as np
import tensorflow as tf

from numpy import pi
from tensorflow import math
from tensorflow_probability import distributions as tfd
from parameters import EIGHT_SCHOOL_K

## ------------------------
## Parameters

two_hills_y = 0.5
two_hills_sigma2 = 0.1

banana_mu = np.array([0.5,0.5], dtype='float32')
banana_cov = np.array([[0.06,0.055],[0.055,0.06]], dtype='float32')
banana_scale = tf.linalg.cholesky(banana_cov)

figure_eight_mu1 = 1 * np.array([-1,-1], dtype='float32')
figure_eight_mu2 = 1 * np.array([1,1], dtype='float32')
figure_eight_scale = 0.45 * np.array([1,1], dtype='float32')
figure_eight_cov = 0.45 * np.array([[1,0],[0,1]], dtype='float32')
figure_eight_pi = 0.5

# source http://www.stat.columbia.edu/~gelman/book/BDA3.pdf p120
eight_schools_y = np.array([28,8,-3,7,-1,1,18,12], dtype='float32')
eight_schools_sigma = np.array([15,10,16,11,9,11,10,18], dtype='float32')
eight_schools_K = EIGHT_SCHOOL_K
eight_schools_replicate = tf.ones([eight_schools_K, 1])


## ------------------------
## Helpers function

def w1(z):
    z1 = z[:, 0]
    return math.sin(2*pi*z1 / 4)

def w2(z):
    z1 = z[:, 0]
    exp_arg = -0.5 * ((z1 - 1) / 0.6) ** 2
    return 3 * math.exp(exp_arg)

def w3(z):
    z1 = z[:, 0]
    return 3 * math.sigmoid((z1 - 1) / 0.3)


## ------------------------
## LOG_JOINT (Likelihood * Prior) distributions

def two_hills_log_pdf(z):

    likelihood = tfd.Normal(loc=z**2, scale=math.sqrt(two_hills_sigma2))
    prior = tfd.Normal(loc=0, scale=1)

    return likelihood.log_prob(two_hills_y) + prior.log_prob(z)

def banana_log_pdf(z):

    z1, z2 = z[:, 0], z[:, 1]
    p = tfd.MultivariateNormalTriL(loc=banana_mu, scale_tril=banana_scale)
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

def energy_1_log_pdf(z):

    z1, z2 = z[:, 0], z[:, 1]
    norm = (z1**2 + z2**2)**0.5
    exp1 = math.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    exp2 = math.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    u = 0.5 * ((norm - 2) / 0.4) ** 2 - math.log(exp1 + exp2)

    return -u

def energy_2_log_pdf(z):

    z2 = z[:, 1]
    return - 0.5 * ((z2 - w1(z)) / 0.4) ** 2

def energy_3_log_pdf(z):

    z2 = z[:, 1]
    exp1 = math.exp(-0.5 * ((z2 - w1(z)) / 0.35) ** 2)
    exp2 = math.exp(-0.5 * ((z2 - w1(z) + w2(z)) / 0.35) ** 2)

    return math.log(exp1 + exp2)

def energy_4_log_pdf(z):

    z2 = z[:, 1]
    exp1 = math.exp(-0.5 * ((z2 - w1(z)) / 0.4) ** 2)
    exp2 = math.exp(-0.5 * ((z2 - w1(z) + w3(z)) / 0.35) ** 2)

    return math.log(exp1 + exp2)

def figure_eight_log_pdf(z):

    comp1 = tfd.MultivariateNormalDiag(loc=figure_eight_mu1, scale_diag=figure_eight_scale)
    comp2 = tfd.MultivariateNormalDiag(loc=figure_eight_mu2, scale_diag=figure_eight_scale)

    return math.log((1-figure_eight_pi)*comp1.prob(z) + figure_eight_pi*comp2.prob(z))

def eight_schools_log_pdf(z):

    # shapes, thetas=(8,N), mu=(N,), tau=(N,)
    thetas, mu, log_tau  = z[:, 0:eight_schools_K], z[:, -2], z[:, -1]

    mu = tf.transpose(eight_schools_replicate * mu)
    log_tau = tf.transpose(eight_schools_replicate * log_tau)

    likelihood = tfd.Normal(loc=thetas, scale=eight_schools_sigma[0:eight_schools_K])
    prior_theta = tfd.Normal(loc=mu, scale=math.exp(log_tau))
    prior_mu = tfd.Normal(loc=0, scale=5)
    prior_tau = tfd.HalfCauchy(loc=0, scale=5)
    log_det_jac = math.log(math.exp(log_tau)) # kept log(exp()) for mathematical understanding.

    return likelihood.log_prob(eight_schools_y[0:eight_schools_K]) + prior_theta.log_prob(thetas) + prior_mu.log_prob(mu) + prior_tau.log_prob(math.exp(log_tau)) + log_det_jac

    # # shapes, thetas=(8,N), mu=(N,), tau=(N,)
    # thetas_tilde, mu, log_tau  = z[:, 0:eight_schools_K], z[:, -2], z[:, -1]

    # # Adapt size of mu an tau.
    # mu = tf.transpose(eight_schools_replicate * mu)
    # log_tau = tf.transpose(eight_schools_replicate * log_tau)
    # zeros = tf.zeros(mu.shape)
    # ones = tf.ones(log_tau.shape)

    # thetas = mu + thetas_tilde * math.exp(log_tau)

    # likelihood = tfd.Normal(loc=thetas, scale=eight_schools_sigma[0:eight_schools_K])
    # prior_theta = tfd.Normal(loc=zeros, scale=ones)
    # prior_mu = tfd.Normal(loc=0, scale=5)
    # prior_tau = tfd.HalfCauchy(loc=0, scale=5)
    # log_det_jac = math.log(math.exp(log_tau)) # kept log(exp()) for mathematical understanding.

    # return likelihood.log_prob(eight_schools_y[0:eight_schools_K]) + prior_theta.log_prob(thetas_tilde) + prior_mu.log_prob(mu) + prior_tau.log_prob(math.exp(log_tau)) + log_det_jac

def get_log_joint_pdf(target_distribution):

    if target_distribution == 'two_hills':
        log_joint_pdf = two_hills_log_pdf
    elif target_distribution == 'banana':
        log_joint_pdf = banana_log_pdf
    elif target_distribution == 'circle':
        log_joint_pdf = circle_log_pdf
    elif target_distribution == 'energy_1':
        log_joint_pdf = energy_1_log_pdf
    elif target_distribution == 'energy_2':
        log_joint_pdf = energy_2_log_pdf
    elif target_distribution == 'energy_3':
        log_joint_pdf = energy_3_log_pdf
    elif target_distribution == 'energy_4':
        log_joint_pdf = energy_4_log_pdf
    elif target_distribution == 'figure_eight':
        log_joint_pdf = figure_eight_log_pdf
    elif target_distribution == 'eight_schools':
        log_joint_pdf = eight_schools_log_pdf

    return log_joint_pdf

