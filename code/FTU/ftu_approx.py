import numpy as np
from scipy.optimize import brent


def ftu_approx_ratio(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    a, b, c = X.min(), X.mean(), X.max()
    s, _, _, _ = brent(
        lambda s: np.var((X - s)**2),
        brack=(a, b, c),
        full_output=True,
    )
    return float(s), float(np.var(np.abs(X - s)) / X.var())


def ftu_approx_statistics(X):
    s, ratio = ftu_approx_ratio(X)
    return s, ratio * 4


def quantile_approx(alpha, m, n_unif):
    Phi_n = []

    for _ in range(m):
        # Generate uniform points
        rng = np.random.default_rng()
        U_b = rng.uniform(0, 1, n_unif)
        # Double FTU - 1st p-value
        _, Phi = ftu_approx_statistics(U_b)
        Phi_n.append(Phi)

    q = np.quantile(Phi_n, alpha)
    return q


def folding_test_approx(X, alpha):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    assert X.ndim == 1 or X.shape[1] == 1,"X must be a 1d array."
    n = X.shape[0]
    s, ftu_stat = ftu_approx_statistics(X)
    q = quantile_approx(alpha, m=1000, n_unif=n)
    if ftu_stat >= q:
        return {'unimodal': True, 'Phi': ftu_stat, 's': s, 'q': q}
    else:
        return {'unimodal': False, 'Phi': ftu_stat, 's': s, 'q': q}

