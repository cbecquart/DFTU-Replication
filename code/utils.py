import numpy as np


def generate_gaussian_mixture(mu, eps, sigma, n):
    groups = []
    for i, mu_i in enumerate(mu):
        groups.append(np.random.normal(loc=mu_i, scale=sigma, size=round(n*eps[i])))
    data = np.concatenate(groups)
    return data


def generate_dirac_mixture(mu, eps, n):
    groups = []
    for i, mu_i in enumerate(mu):
        groups.append(np.random.normal(loc=mu_i, scale=0, size=round(n*eps[i])))
    data = np.concatenate(groups)
    return data


def generate_gauss_unif_mixture(gauss_mu, unif_lbound, unif_ubound, eps, sigma, n):
    groups = []
    groups.append(np.random.normal(loc=gauss_mu, scale=sigma, size=round(n*eps[0])))
    groups.append(np.random.uniform(low=unif_lbound, high=unif_ubound, size=round(n*eps[1])))
    data = np.concatenate(groups)
    return data
