"""
In order to prove that the DFTU never fails in the case of a mixture of 3 Dirac distributions, we want to prove that
new_F > new_x2minimize. To do so, we show that min (new_F - new_x2) > 0.
"""


import numpy as np
from scipy.optimize import minimize


def compute_eps2(eps1, eps3):
    return 1 - eps1 - eps3


def compute_bounds_x1(eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    lower = (-np.sqrt(3)/2) * np.sqrt((1 - eps1) / eps1)
    upper_case1 = - np.sqrt(eps3 / (1 - eps3)) * (2*eps2 + 4*eps1*eps3) / np.sqrt(4*eps1*eps3*(eps2 + 4*eps1*eps3))
    upper_case2 = (-np.sqrt(2)/2) * np.sqrt((1 - eps1)/eps1 + np.sqrt((1 - eps1)/eps1) * np.sqrt(eps3 / (1 - eps3)))
    upper = max(upper_case1, upper_case2)
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


def compute_x_tilde(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    approx_pivot = compute_gamma(x1, eps1, eps3) / 2
    x1_tilde = np.abs(x1 - approx_pivot)
    x2_tilde = np.abs(x2 - approx_pivot)
    x3_tilde = np.abs(x3 - approx_pivot)
    return x1_tilde, x2_tilde, x3_tilde


def compute_new_exp(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    x1_tilde, x2_tilde, x3_tilde = compute_x_tilde(x1, eps1, eps3)
    return eps1 * x1_tilde + eps2 * x2_tilde + eps3 * x3_tilde


def compute_new_var(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    x1_tilde, x2_tilde, x3_tilde = compute_x_tilde(x1, eps1, eps3)
    new_exp = compute_new_exp(x1, eps1, eps3)
    if np.isnan(new_exp):
        return np.nan
    return eps1 * x1_tilde**2 + eps2 * x2_tilde**2 + eps3 * x3_tilde**2 - new_exp**2


def compute_new_x2(x1, eps1, eps3):
    x2 = compute_x2(x1, eps1, eps3)
    approx_pivot = compute_gamma(x1, eps1, eps3) / 2
    x2_tilde = np.abs(x2 - approx_pivot)
    new_exp = compute_new_exp(x1, eps1, eps3)
    new_var = compute_new_var(x1, eps1, eps3)
    if new_var <= 0 or any(np.isnan(val) for val in [x2, new_exp, new_var]):
        return np.nan
    return (x2_tilde - new_exp) / np.sqrt(new_var)


def compute_new_x(x1, eps1, eps3):
    x1_tilde, x2_tilde, x3_tilde = compute_x_tilde(x1, eps1, eps3)
    new_exp = compute_new_exp(x1, eps1, eps3)
    new_var = compute_new_var(x1, eps1, eps3)
    if new_var <= 0 or any(np.isnan(val) for val in [new_exp, new_var]):
        return np.nan
    new_x1 = (x1_tilde - new_exp) / np.sqrt(new_var)
    new_x2 = (x2_tilde - new_exp) / np.sqrt(new_var)
    new_x3 = (x3_tilde - new_exp) / np.sqrt(new_var)
    return new_x1, new_x2, new_x3


def compute_new_F(eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    if eps2 <= 0 or eps2 >= 1:
        return np.nan
    return (-np.sqrt(3)/2) * np.sqrt((1 - eps2) / eps2)


def compute_new_U(x1, eps1, eps3):
    eps2 = compute_eps2(eps1, eps3)
    if eps2 <= 0 or eps2 >= 1:
        return np.nan
    new_x1, new_x2, new_x3 = compute_new_x(x1, eps1, eps3)
    if new_x1 > new_x3:
        _, upper = compute_bounds_x1(eps2, eps1)
    elif new_x1 < new_x3:
        _, upper = compute_bounds_x1(eps2, eps3)
    else:
        return np.nan

    return upper


# Different objective functions to be minimized

def objective_nofail(vars):
    x1, eps1, eps3 = vars
    new_x2 = compute_new_x2(x1, eps1, eps3)
    new_F = compute_new_F(eps1, eps3)
    if np.isnan(new_x2) or np.isnan(new_F):
        return 1e6
    return new_F - new_x2


def objective_gamma_l(vars):
    x1, eps1, eps3 = vars
    gamma = compute_gamma(x1, eps1, eps3)
    x2 = compute_x2(x1, eps1, eps3)
    if np.isnan(gamma) or np.isnan(x2):
        return 1e6
    return gamma - x1 - x2


def objective_gamma_u(vars):
    x1, eps1, eps3 = vars
    gamma = compute_gamma(x1, eps1, eps3)
    x2 = compute_x2(x1, eps1, eps3)
    x3 = compute_x3(x1, eps1, eps3)
    if np.isnan(gamma) or np.isnan(x2):
        return 1e6
    return x2 + x3 - gamma


def objective_res4(vars):
    x1, eps1, eps3 = vars
    if not (0 < eps1 < 1 and 0 < eps3 < 1 - eps1):
        return 1e6  # invalid
    new_x2 = compute_new_x2(x1, eps1, eps3)
    U = compute_new_U(x1, eps1, eps3)
    if np.isnan(new_x2) or np.isnan(U):
        return 1e6
    return new_x2**2 - U**2


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


constraints = (
    {'type': 'ineq', 'fun': constraint_eps3},
    {'type': 'ineq', 'fun': constraint_x_lower},
    {'type': 'ineq', 'fun': constraint_x_upper}
)


# Bounds
bounds = [
    (-10, 0),  # x1
    (0.01, 0.99),  # eps1
    (0.01, 0.99)  # eps3
]


# Initial point
x0 = [-1.203, 0.34, 0.32]


# Optimisation

# Check that new_x2 < new_F <=> test does not fail under the conditions:
# i) new_x2 is the first group and ii) the second-step pivot is in the second-step first interval (Proposition 3.1)

# i) new_x2 is the first group <=> gamma/2 > (x1 + x2) / 2 and gamma/2 < (x2 + x3) / 2

# Check that gamma > x1 + x2

result = minimize(objective_gamma_l, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(gamma - x1 - x2):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  gamma - x1 - x2 = {objective_gamma_l(result.x)}")
else:
    print("❌ Optimization failed:", result.message)

# Check that gamma < x2 + x3

result = minimize(objective_gamma_u, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(gamma/2 - x2):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  x2 + x3 - gamma = {objective_gamma_u(result.x)}")
else:
    print("❌ Optimization failed:", result.message)


# ii) the second-step pivot is in the second-step first interval (Proposition 3.1)

result = minimize(objective_res4, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x1_opt, eps1_opt, eps3_opt = result.x
    print(f"✅ Solution found for min(new_x2**2 - U**2):")
    print(f"  x1 = {x1_opt}")
    print(f"  eps1 = {eps1_opt}")
    print(f"  eps3 = {eps3_opt}")
    print(f"  new_x2**2 - U**2 = {objective_res4(result.x)}")
else:
    print("❌ Optimization failed:", result.message)

# Proposition 3.2 applied to the second step

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
