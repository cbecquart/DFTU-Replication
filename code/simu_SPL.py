"""
Simulations used in the SPL paper. The FTU*, FTU**, DFTU and dip-test are computed on different univariate mixtures.
"""
import numpy as np
import pandas as pd
from FTU.ftu_exact import folding_test_exact
from FTU.ftu_approx import folding_test_approx
from FTU.double_ftu import double_folding_test
from utils import generate_gaussian_mixture, generate_dirac_mixture, generate_gauss_unif_mixture
from diptest import diptest
# import timeit


# Parameters
dict_GMM1 = {
    "index": "GMM1",
    "model_label": "GMM",
    "mu": np.array([0]),
    "eps": np.array([1]),
    "sigma": 1
}

dict_GMM2 = {
    "index": "GMM2",
    "model_label": "GMM",
    "mu": np.array([0, 1]),
    "eps": np.array([0.5, 0.5]),
    "sigma": 0.5
}

dict_GUM1 = {
    "index": "GUM1",
    "model_label": "GUM",
    "gauss_mu": 0,
    "unif_lbound": 4,
    "unif_ubound": 8,
    "eps": np.array([0.6, 0.4]),
    "sigma": 0.5
}

dict_GUM2 = {
    "index": "GUM2",
    "model_label": "GUM",
    "gauss_mu": 0,
    "unif_lbound": 1,
    "unif_ubound": 4,
    "eps": np.array([0.6, 0.4]),
    "sigma": 0.5
}

dict_DMM1 = {
    "index": "DMM1",
    "model_label": "DMM",
    "mu": np.array([-2, 0, 2]),
    "eps": np.array([1/3, 1/3, 1/3])
}

dict_GMM3 = {
    "index": "GMM3",
    "model_label": "GMM",
    "mu": np.array([-2, 0, 2]),
    "eps": np.array([1/3, 1/3, 1/3]),
    "sigma": 0.5
}

dict_DMM2 = {
    "index": "DMM2",
    "model_label": "DMM",
    "mu": np.array([-3, -1.5, 2.5, 4, 11]),
    "eps": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
}

dict_GMM4 = {
    "index": "GMM4",
    "model_label": "GMM",
    "mu": np.array([-3, -1.5, 2.5, 4, 11]),
    "eps": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    "sigma": 0.5
}

list_models = [dict_GMM1, dict_GMM2, dict_GUM1, dict_GUM2, dict_DMM1, dict_GMM3, dict_DMM2, dict_GMM4]

N_RUNS = 100
N_DATA = 1000


def run_data_iter(data, ftu_approx_count_unimodal, ftu_raw_count_unimodal, dftu_count_unimodal, diptest_count_unimodal):

    # FTU - approx pivot
    res_ftu_approx = folding_test_approx(data, alpha=0.05)
    ftu_approx_count_unimodal += (res_ftu_approx["unimodal"])

    # FTU - best pivot
    res_ftu_raw = folding_test_exact(data, alpha=0.05)
    ftu_raw_count_unimodal += (res_ftu_raw["unimodal"])

    # DFTU
    res_dftu = double_folding_test(data, alpha=0.05, alpha_1=0.03)
    dftu_count_unimodal += res_dftu

    # diptest
    _, pv_diptest = diptest(data)
    diptest_count_unimodal += (pv_diptest >= 0.05)

    return ftu_approx_count_unimodal, ftu_raw_count_unimodal, dftu_count_unimodal, diptest_count_unimodal


def simulation_iter(data_gen_func, dict_param, n_runs):

    ftu_approx_count_unimodal = 0
    ftu_raw_count_unimodal = 0
    dftu_count_unimodal = 0
    diptest_count_unimodal = 0


    for _ in range(n_runs):
        data = data_gen_func(**dict_param)
        (ftu_approx_count_unimodal, ftu_raw_count_unimodal, dftu_count_unimodal,
         diptest_count_unimodal) = run_data_iter(data, ftu_approx_count_unimodal, ftu_raw_count_unimodal,
                                                 dftu_count_unimodal, diptest_count_unimodal)

    return ftu_approx_count_unimodal, ftu_raw_count_unimodal, dftu_count_unimodal, diptest_count_unimodal


if __name__ == "__main__":
    results = []

    for model in list_models:
        eps = model["eps"]
        model_label = model["model_label"]

        assert model_label in ['GMM', 'DMM', 'GUM'], "model must be one of ['GMM', 'DMM', 'GUM']"
        if model_label == 'GMM':
            data_gen_func = generate_gaussian_mixture
            dict_param = {"mu": model["mu"], "eps": eps, "sigma": model["sigma"], "n": N_DATA}
        elif model_label == 'DMM':
            data_gen_func = generate_dirac_mixture
            dict_param = {"mu": model["mu"], "eps": eps, "n": N_DATA}
        else:
            data_gen_func = generate_gauss_unif_mixture
            dict_param = {"gauss_mu": model["gauss_mu"], "unif_lbound": model["unif_lbound"],
                          "unif_ubound": model["unif_ubound"], "eps": eps, "sigma": model["sigma"], "n": N_DATA}

        (ftu_approx_count_unimodal, ftu_raw_count_unimodal,
         dftu_count_unimodal, diptest_count_unimodal) = simulation_iter(data_gen_func, dict_param, n_runs=N_RUNS)

        results.append({
            "model": model["index"],
            "ftu_approx": ftu_approx_count_unimodal,
            "ftu_raw": ftu_raw_count_unimodal,
            "dftu": dftu_count_unimodal,
            "diptest": diptest_count_unimodal
        })

        df_results = pd.DataFrame(results)
        df_results.to_csv("../results/results_simu.csv", index=False)

