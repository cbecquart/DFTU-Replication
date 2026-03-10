"""
In order to prove that the DFTU never fails in the case of a mixture of 3 Dirac distributions, we want to prove that
new_F > new_x2minimize. To do so, we show that min (new_F - new_x2) > 0. See Zotero 3_groupes_double_pliage.pdf for the
details of the computations.
"""


import numpy as np
from scipy.optimize import minimize


def compute_eps2(eps1, eps3):
    return 1 - eps1 - eps3


def compute_bounds_x1(eps1, eps3):
    lower = (-np.sqrt(3)/2) * np.sqrt((1 - eps1) / eps1)
    upper = (-np.sqrt(2)/2) * np.sqrt((1 - eps1)/eps1 + np.sqrt((1 - eps1)/eps1) * np.sqrt(eps3 / (1 - eps3)))
    return lower, upper


def compute_x2(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    expr = 1 - eps1 * (1 + x1**2)
    if expr < 0 or eps2 <= 0:
        return np.nan
    return (-eps1 / (1 - eps1)) * x1 - (1 / (1 - eps1)) * np.sqrt(eps3 / eps2) * np.sqrt(expr)


def compute_x3(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    x2 = compute_x2(x1, eps1, eps3)
    if np.isnan(x2):
        return np.nan
    return (-1 / eps3) * (eps1 * x1 + eps2 * x2)


def compute_gamma(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    eps2 = compute_eps2(eps1, eps3)
    if np.isnan(x2) or np.isnan(x3):
        return np.nan
    return eps1 * x1 ** 3 + eps2 * x2 ** 3 + eps3 * x3 ** 3


def compute_new_exp(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    eps2 = compute_eps2(eps1, eps3)
    gamma = eps1 * x1 ** 3 + eps2 * x2 ** 3 + eps3 * x3 ** 3
    if np.isnan(x2) or np.isnan(x3):
        return np.nan
    return eps1 * (-x1 + gamma) + eps2 * x2 + eps3 * x3


def compute_new_var(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    new_exp = compute_new_exp(x1, eps1, eps3)
    if any(np.isnan(val) for val in [x2, x3, new_exp]):
        return np.nan
    eps2 = compute_eps2(eps1, eps3)
    gamma = eps1 * x1 ** 3 + eps2 * x2 ** 3 + eps3 * x3 ** 3
    return eps1 * (-x1 + gamma)**2 + eps2 * x2**2 + eps3 * x3**2 - new_exp**2


def compute_new_x2(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    new_exp = compute_new_exp(x1, eps1, eps3)
    new_var = compute_new_var(x1, eps1, eps3)
    if new_var <= 0 or any(np.isnan(val) for val in [x2, new_exp, new_var]):
        return np.nan
    return (x2 - new_exp) / np.sqrt(new_var)


def compute_new_x(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    new_exp = compute_new_exp(x1, eps1, eps3)
    new_var = compute_new_var(x1, eps1, eps3)
    if new_var <= 0 or any(np.isnan(val) for val in [x2, new_exp, new_var]):
        return np.nan
    new_x1 = (-x1 + compute_gamma(x1, eps1, eps3) - new_exp) / np.sqrt(new_var)
    new_x2 = (x2 - new_exp) / np.sqrt(new_var)
    new_x3 = (x3 - new_exp) / np.sqrt(new_var)
    return new_x1, new_x2, new_x3


def compute_new_F(eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    if eps2 <= 0 or eps2 >= 1:
        return np.nan
    return (-np.sqrt(3)/2) * np.sqrt((1 - eps2) / eps2)


def compute_J(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    if eps2 <= 0 or eps2 >= 1:
        return np.nan
    new_x1, new_x2, new_x3 = compute_new_x(x1, eps1, eps3)
    if new_x1 > new_x3:
        c = eps1
        b = eps3
    elif new_x1 < new_x3:
        c = eps3
        b = eps1
    else:
        return np.nan
    return (1-eps2)/eps2 - 2 * new_x2 * np.sqrt((c/b) * (1 - eps2*(1 + new_x2**2)))


# Different objective functions to be minimized
def objective_nofail(vars):
    x1, eps1, eps3 = vars
    if not (0 < eps1 < 1 and 0 < eps3 < 1 - eps1):
        return 1e6  # invalid
    new_x2 = compute_new_x2(x1, eps1, eps3)
    new_F = compute_new_F(eps1, eps3)
    if np.isnan(new_x2) or np.isnan(new_F):
        return 1e6
    return new_F - new_x2


def objective_gamma(vars):
    x1, eps1, eps3 = vars
    if not (0 < eps1 < 1 and 0 < eps3 < 1 - eps1):
        return 1e6  # invalid
    gamma = compute_gamma(x1, eps1, eps3)
    x2 = compute_x2(x1, eps1, eps3)
    if np.isnan(gamma) or np.isnan(x2):
        return 1e6
    return gamma - x1 - x2


def objective_res4(vars):
    x1, eps1, eps3 = vars
    if not (0 < eps1 < 1 and 0 < eps3 < 1 - eps1):
        return 1e6  # invalid
    new_x2 = compute_new_x2(x1, eps1, eps3)
    J = compute_J(x1, eps1, eps3)
    if np.isnan(new_x2) or np.isnan(J):
        return 1e6
    return 2 * (new_x2**2) - J


# Constraints

def constraint_eps3(vars):
    x1, eps1, eps3 = vars
    return 1 - eps1 - eps3


def constraint_eps2(vars):
    x1, eps1, eps3 = vars
    return compute_eps2(eps1, eps3) - eps1 * eps3 / 3


def constraint_x_lower(vars):
    x1, eps1, eps3 = vars
    lower, _ = compute_bounds_x1(eps1, eps3)
    return x1 - lower


def constraint_x_upper(vars):
    x1, eps1, eps3 = vars
    _, upper = compute_bounds_x1(eps1, eps3)
    return upper - x1


constraints = [
    {'type': 'ineq', 'fun': constraint_eps3},
    {'type': 'ineq', 'fun': constraint_eps2},
    {'type': 'ineq', 'fun': constraint_x_lower},
    {'type': 'ineq', 'fun': constraint_x_upper}
]


# Bounds
bounds = [
    (-10, 0),  # x1 (on restreint à des bornes larges, les vraies sont imposées via contraintes)
    (0.01, 0.99),  # eps1
    (0.01, 0.99)  # eps3
]


# Initial point
x0 = [-1.203, 0.34, 0.32]
#x0 = [-2.5, 0.1, 0.1]


# Optimisation

# Check that new_x2 < new_F <=> test does not fail under the conditions:
# i) new_x2 is the first group and ii) the new pivot is in the new first interval (see condition of Result 4)

result = minimize(objective_nofail, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={"maxiter": 5000, "ftol": 1e-8})

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(new_F - new_x2):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  new_F - new_x2 = {objective_nofail(result.x)}")
else:
    print("❌ Optimization failed:", result.message)


# Check that gamma > x1 + x2 <=> new_x2 is the first group center

result = minimize(objective_gamma, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(gamma - x1 - x2):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  gamma - x1 - x2 = {objective_nofail(result.x)}")
else:
    print("❌ Optimization failed:", result.message)


# Check that the condition in Result 4 still holds after folding
# todo: what if some new group centers are equal ?
# todo: compute_J: use the formula from mathematica or values of new_x1 and new_x2

result = minimize(objective_res4, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(2 * (new_x2**2) - J):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  2 * (new_x2**2) - J = {objective_nofail(result.x)}")
else:
    print("❌ Optimization failed:", result.message)
