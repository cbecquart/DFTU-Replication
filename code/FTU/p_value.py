import numpy as np
from ftu_non_approx import folding_test_raw
from ftu_Rmigration import folding_test2
# from FTU.ftu_non_approx import folding_test_raw
# from FTU.ftu_Rmigration import folding_test2

m = 10**3

alpha = 0.05
alpha_1 = 0.03
alpha_2 = (alpha - alpha_1) / (1 - alpha_1)
print(alpha_2)

n = 1000

# Compute q1
Phi_n_step1 = []

for _ in range(m):
    # Generate uniform points
    rng = np.random.default_rng()
    U_b = rng.uniform(0, 1, n)
    U_b = np.asarray(U_b, dtype=np.float64).reshape(-1, 1)
    # Double FTU - 1st p-value
    results_ftu_raw = folding_test_raw(U_b)
    Phi_n_step1.append(results_ftu_raw["Phi"])

q1 = np.quantile(Phi_n_step1, alpha_1)

# Compute q2
Phi_n_step2 = []

for _ in range(m):
    # Generate uniform points
    rng = np.random.default_rng()
    U_b = rng.uniform(0, 1, n)
    U_b = np.asarray(U_b, dtype=np.float64).reshape(-1, 1)

    # Step 1
    results_ftu_raw = folding_test_raw(U_b)
    # Step 2
    if results_ftu_raw["Phi"] >= q1:
        results_ftu_approx = folding_test2(U_b)
        x_new = np.abs(U_b - results_ftu_approx['s2star'])
        results_ftu_double = folding_test_raw(x_new)
        Phi_n_step2.append(results_ftu_double["Phi"])

q2 = np.quantile(Phi_n_step2, alpha_2)

print(alpha_1, alpha_2)
print(q1)
print(q2)
