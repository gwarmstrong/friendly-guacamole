import numpy as np
from scipy.stats import norm
from numpy.random import poisson, lognormal
from skbio.stats.composition import closure


def chain_interactions(gradient, mu, sigma):
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma[i])
          for i in range(len(mu))]
    return np.vstack(xs)


def resample_counts(X, depth, kappa=1):
    mu = depth * closure(X)
    n_samples = len(X)
    new_samples = np.vstack(
        [poisson(lognormal(np.log(mu[i, :]), kappa))
         for i in range(n_samples)
         ]
    )
    return new_samples
