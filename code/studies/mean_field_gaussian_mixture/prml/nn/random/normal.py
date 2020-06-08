import numpy as np
from prml.nn.array.array import asarray
from scipy.stats import truncnorm


def normal(mean, std, size):
    return asarray(np.random.normal(mean, std, size))


def truncnormal(min, max, scale, size):
    return asarray(truncnorm(a=min, b=max, scale=scale).rvs(size))
