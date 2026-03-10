import numpy as np
from FTU.ftu_exact import ftu_exact_statistics
from FTU.ftu_approx import ftu_approx_statistics


def double_folding_test(x, alpha=0.05, alpha_1 = 0.04):

    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = x.shape[0]

    _, Phi_exact = ftu_exact_statistics(x)
    # s_exact, Phi_exact = ftu_exact_statistics(x)
    q1 = quantile_step1(alpha_1, m=1000, n_unif=n)

    if (Phi_exact >= q1):
        s_approx, _ = ftu_approx_statistics(x)
        # s_approx = s_exact - 0.1 * x.var()
        x_new = np.abs(x - s_approx)
        _, Phi_exact_step2 = ftu_exact_statistics(x_new)

        # critical region
        alpha_2 = (alpha - alpha_1) / (1 - alpha_1)
        q2 = quantile_step2(alpha_2, q1, m=1000, n_unif=n)

        if (Phi_exact_step2 < q2):
            unimodal = False
        else:
            unimodal = True
    else:
        unimodal = False
    return unimodal


def quantile_step1(alpha_1, m, n_unif):
    Phi_n_step1 = []

    for _ in range(m):
        # Generate uniform points
        rng = np.random.default_rng()
        U_b = rng.uniform(0, 1, n_unif)
        # Double FTU - 1st p-value
        _, Phi_exact = ftu_exact_statistics(U_b)
        Phi_n_step1.append(Phi_exact)

    q1 = np.quantile(Phi_n_step1, alpha_1)
    return q1


def quantile_step2(alpha_2, q1, m, n_unif):
    Phi_n_step2 = []

    for _ in range(m):
        # Generate uniform points
        rng = np.random.default_rng()
        U_b = rng.uniform(0, 1, n_unif)

        # Step 1
        _, Phi_exact = ftu_exact_statistics(U_b)
        # Step 2
        if Phi_exact >= q1:
            s_approx, _ = ftu_approx_statistics(U_b)
            x_new = np.abs(U_b - s_approx)
            _, Phi_exact_step2 = ftu_exact_statistics(x_new)
            Phi_n_step2.append(Phi_exact_step2)

    q2 = np.quantile(Phi_n_step2, alpha_2)

    return q2

