"""
Deeper analysis of the Gaussian case. Do we recover the results obtained for Dirac mixtures if the noise is small ?
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FTU.ftu_exact import folding_test_exact
from FTU.ftu_approx import folding_test_approx
from FTU.double_ftu import double_folding_test
from utils import generate_gaussian_mixture
from diptest import diptest


def get_results(mu, eps, sigma):
    data = generate_gaussian_mixture(mu=mu, eps=eps, sigma=sigma, n=10000)
    results_ftu_exact = folding_test_exact(data, alpha=0.05)
    ftu_exact_uni = "Uni" if results_ftu_exact["unimodal"] else "Multi"
    Phi_exact = np.round(results_ftu_exact["Phi"], 2)
    results_ftu_approx = folding_test_approx(data, alpha=0.05)
    ftu_approx_uni = "Uni" if results_ftu_approx["unimodal"] else "Multi"
    Phi_approx = np.round(results_ftu_approx["Phi"], 2)
    results_dftu = double_folding_test(data, alpha=0.05, alpha_1=0.03)
    dftu_uni = "Uni" if results_dftu else "Multi"
    _, pv_diptest = diptest(data)
    diptest_uni = "Uni" if (pv_diptest > 0.05) else "Multi"
    return data, Phi_exact, Phi_approx, ftu_exact_uni, ftu_approx_uni, dftu_uni, diptest_uni


def plot_results(mu, eps, sigma1, sigma2, sigma3, i):
    data1, Phi_exact1, Phi_approx1, ftu_exact_uni1, ftu_approx_uni1, dftu_uni1, diptest_uni1 = get_results(mu, eps, sigma1)
    data2, Phi_exact2, Phi_approx2, ftu_exact_uni2, ftu_approx_uni2, dftu_uni2, diptest_uni2 = get_results(mu, eps, sigma2)
    data3, Phi_exact3, Phi_approx3, ftu_exact_uni3, ftu_approx_uni3, dftu_uni3, diptest_uni3 = get_results(mu, eps, sigma3)

    title1 = (f"σ² = {np.round(sigma1 ** 2, 5)}, Φ* = {Phi_exact1}, Φ** = {Phi_approx1}, <br> "
              f"FTU*: {ftu_exact_uni1}, FTU**: {ftu_approx_uni1} <br> DFTU: {dftu_uni1}, dip test: {diptest_uni1}")
    title2 = (f"σ² = {np.round(sigma2 ** 2, 5)}, Φ* = {Phi_exact2}, Φ** = {Phi_approx2}, <br> "
              f"FTU*: {ftu_exact_uni2}, FTU**: {ftu_approx_uni2} <br> DFTU: {dftu_uni2}, dip test: {diptest_uni2}")
    title3 = (f"σ² = {np.round(sigma3 ** 2, 5)}, Φ* = {Phi_exact3}, Φ** = {Phi_approx3}, <br> "
              f"FTU*: {ftu_exact_uni3}, FTU**: {ftu_approx_uni3} <br> DFTU: {dftu_uni3}, dip test: {diptest_uni3}")

    if i==2:
        title1 = (f"σ² = {np.round(sigma1 ** 2, 5)}, Φ* = <span style='color:magenta'>{Phi_exact1}</span>, Φ** = {Phi_approx1}, <br> "
                  f"FTU*: {ftu_exact_uni1}, FTU**: {ftu_approx_uni1} <br> DFTU: {dftu_uni1}, dip test: {diptest_uni1}")
        title2 = (f"σ² = {np.round(sigma2 ** 2, 5)}, Φ* = <span style='color:blue'>{Phi_exact2}</span>, Φ** = {Phi_approx2}, <br> "
                  f"FTU*: {ftu_exact_uni2}, FTU**: {ftu_approx_uni2} <br> DFTU: {dftu_uni2}, dip test: {diptest_uni2}")
        title3 = (f"σ² = {np.round(sigma3 ** 2, 5)}, Φ* = <span style='color:blue'>{Phi_exact3}</span>, Φ** = {Phi_approx3}, <br> "
                  f"FTU*: {ftu_exact_uni3}, FTU**: {ftu_approx_uni3} <br> DFTU: {dftu_uni3}, dip test: {diptest_uni3}")


    titles = (title1, title2, title3)

    fig = make_subplots(rows=1, cols=3, start_cell="top-left", subplot_titles=titles)
    fig.add_trace(go.Histogram(x=data1, marker_color="lightblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=data2, marker_color="lightblue"), row=1, col=2)
    fig.add_trace(go.Histogram(x=data3, marker_color="lightblue"), row=1, col=3)

    fig.update_layout(width=1400, height=320, font=dict(size=19),
                      template="plotly_white", showlegend=False)
    fig.update_annotations(font_size=22)
    fig.write_image(f"../results/plots_gaussianSPL/gaussian_details{i}.png")


# Case 1: 3 groups - Both fail
sigma1 = 0.15
sigma2 = 0.25
sigma3 = 0.5

mu = np.array([-2, 0, 2])
eps = np.array([1/3, 1/3, 1/3])

plot_results(mu, eps, sigma1, sigma2, sigma3, i=1)


# Case 2: 3 groups - FTU** fails
mu = np.array([-2, 0, 2])
eps = np.array([0.2, 0.4, 0.4])
plot_results(mu, eps, sigma1, sigma2, sigma3, i=2)


# Case 3: 3 groups - Both works
mu = np.array([-2, 0, 2.5])
eps = np.array([0.5, 0.2, 0.3])

plot_results(mu, eps, sigma1, sigma2, sigma3, i=3)

