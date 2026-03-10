"""
Explore a mixture of 2 Gaussian distributions, in particular the impact of the parameter sigma. Plot the behaviour of
the folding statistic, the pivot s^*, the cdf and pdf of the distribution.

It is important to center the data for the formula of the function f. Var=1 or B=1 is not necessary, just be careful to
update the formula of the variance in create_points_for_given_sigma().
"""


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm


# Mixture parameters
eps1 = 0.3
eps2 = 1 - eps1

mu1 = -2.8
mu2 = 1.2

mu = np.array([mu1, mu2])
eps = np.array([eps1, eps2])


# Distribution parameters
sigma1 = np.sqrt(0.01)
sigma2 = np.sqrt(0.8)
sigma3 = np.sqrt(1.34)
sigma4 = np.sqrt(7)


def centering_step(mu, eps, sigma):
    """
    Center the data, and order the group means. This step is required for the computations.
    """
    mu = (mu - np.average(mu, weights=eps))
    idx = mu.argsort()
    mu = mu[idx]
    eps = eps[idx]

    return mu, eps, sigma


def f(s, mu, sigma):
    z = (s - mu) / sigma
    return (s - mu) * (2 * norm.cdf(z) - 1) + 2 * sigma * norm.pdf(z)


def create_points_for_given_sigma(mu, eps, sigma):
    mu, eps, sigma = centering_step(mu, eps, sigma)
    s_values = np.linspace(mu[0]-0.25, mu[1]+0.25, 1000)

    var_x = np.average(mu ** 2, weights=eps) - np.average(mu, weights=eps) ** 2 + sigma ** 2
    var_values = var_x + s_values**2 - (eps[0] * f(s_values, mu[0], sigma) + eps[1] * f(s_values, mu[1], sigma))**2
    Phi_values = 4 * var_values / var_x

    return Phi_values, s_values


def plot_ftu_and_pivot_vs_sigma_sq():
    """
    Create a figure with two subplots: the first is the exact FTU statistic Phi* as a function of sigma^2,
    the second is the exact pivot s* as a function of sigma^2.
    """

    # Range of sigma^2: x-axis values
    sigma_sq_arr = np.linspace(0.01, 7, 1000)
    sigma_arr = np.sqrt(sigma_sq_arr)

    # Compute Phi* and s*: y-axis values
    Phi_star = []
    s_star = []
    for sigma in sigma_arr:
        Phi_values, s_values = create_points_for_given_sigma(mu, eps, sigma)
        idx = np.argmin(Phi_values)
        Phi_star.append(Phi_values[idx])
        s_star.append(s_values[idx])

    # Plot
    fig = make_subplots(rows=1, cols=1, start_cell="top-left", x_title="σ²")
    fig.add_trace(
        go.Scatter(x=sigma_sq_arr, y=np.ones(shape=sigma_sq_arr.shape), mode='markers', line=dict(color='red')),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=sigma_sq_arr, y=Phi_star, mode='markers', line=dict(color='black')),
                  row=1, col=1)
    # fig.add_trace(go.Scatter(x=sigma_sq_arr, y=s_star, mode='markers', line=dict(color='black')),
    #               row=2, col=1)
    fig.add_vline(x=sigma1 ** 2, line=dict(color='darkblue', dash='dash'),
                  annotation_text=str(np.round(sigma1 ** 2, 2)),
                  annotation=dict(font=dict(color='darkblue')), annotation_position="bottom")
    fig.add_vline(x=sigma2 ** 2, line=dict(color='#2D68C4', dash='dash'), annotation_text=str(np.round(sigma2 ** 2, 1)),
                  annotation=dict(font=dict(color='#2D68C4')), annotation_position="bottom")
    fig.add_vline(x=sigma3 ** 2, line=dict(color='#2f97ff', dash='dash'), annotation_text=str(np.round(sigma3 ** 2, 2)),
                  annotation=dict(font=dict(color='#2f97ff')), annotation_position="bottom")
    fig.add_vline(x=sigma4 ** 2, line=dict(color='#8FD9FB', dash='dash'), annotation_text=str(np.round(sigma4 ** 2, 1)),
                  annotation=dict(font=dict(color='#8FD9FB')), annotation_position="bottom")

    fig.update_yaxes(title_text="FTU statistic", row=1, col=1)
    # fig.update_yaxes(title_text="Exact pivot", row=2, col=1)
    fig.update_xaxes(tickvals=[0, 1, 2, 3, 4, 5, 6, 7], ticktext=["", "", "", "3", "4", "5", "6", ""], row=1, col=1)
    fig.update_xaxes(tickvals=[0, 1, 2, 3, 4, 5, 6, 7], ticktext=["", "", "", "3", "4", "5", "6", ""], row=2, col=1)
    fig.update_traces(marker=dict(size=3))
    fig.update_annotations(font_size=20)
    fig.update_layout(width=1500, height=320, template="plotly_white", font=dict(size=20), showlegend=False)
    fig.write_image(f"../results/plots_Example2/ftu_vs_sigma_sq.png")


def plot_pdf_2_gaussian(mu1, mu2, eps1, eps2, sigma1, sigma2, sigma3, sigma4):
    """
    Create a figure with 5 subfigures: the density of the distribution for 4 values of sigma, and a vertical line
    representing s*.
    """

    x = np.linspace(-5, 4.5, 1000)

    # Density
    mixture_pdf1 = eps1 * norm.pdf(x, mu1, sigma1) + eps2 * norm.pdf(x, mu2, sigma1)
    mixture_pdf2 = eps1 * norm.pdf(x, mu1, sigma2) + eps2 * norm.pdf(x, mu2, sigma2)
    mixture_pdf3 = eps1 * norm.pdf(x, mu1, sigma3) + eps2 * norm.pdf(x, mu2, sigma3)
    mixture_pdf4 = eps1 * norm.pdf(x, mu1, sigma4) + eps2 * norm.pdf(x, mu2, sigma4)

    # Values of s*
    Phi_values1, s_values1 = create_points_for_given_sigma(mu, eps, sigma=sigma1)
    Phi_values2, s_values2 = create_points_for_given_sigma(mu, eps, sigma=sigma2)
    Phi_values3, s_values3 = create_points_for_given_sigma(mu, eps, sigma=sigma3)
    Phi_values4, s_values4 = create_points_for_given_sigma(mu, eps, sigma=sigma4)

    x_min1 = s_values1[np.argmin(Phi_values1)]
    x_min2 = s_values1[np.argmin(Phi_values2)]
    x_min3 = s_values1[np.argmin(Phi_values3)]
    x_min4 = s_values1[np.argmin(Phi_values4)]

    # Plot

    fig = make_subplots(rows=1, cols=4, start_cell="top-left", x_title="x", y_title="pdf(x)")

    fig.add_trace(
        go.Scatter(x=x, y=mixture_pdf1, mode='lines', name="σ² = " + str(sigma1 ** 2), line=dict(color='darkblue')),
        row=1, col=1)
    fig.add_vline(x=x_min1, line=dict(color='darkblue', dash='dash'), annotation_text="s*",
                  annotation=dict(font=dict(color='darkblue')), annotation_position="top", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=mixture_pdf2, mode='lines', name="σ² = " + str(sigma2 ** 2), line=dict(color='#2D68C4')),
        row=1, col=2)
    fig.add_vline(x=x_min2, line=dict(color='#2D68C4', dash='dash'), annotation_text="s*",
                  annotation=dict(font=dict(color='#2D68C4')), annotation_position="top", row=1, col=2)

    fig.add_trace(
        go.Scatter(x=x, y=mixture_pdf3, mode='lines', name="σ² = " + str(sigma3 ** 2), line=dict(color='#2f97ff')),
        row=1, col=3)
    fig.add_vline(x=x_min3, line=dict(color='#2f97ff', dash='dash'), annotation_text="s*",
                  annotation=dict(font=dict(color='#2f97ff')), annotation_position="top", row=1, col=3)

    fig.add_trace(
        go.Scatter(x=x, y=mixture_pdf4, mode='lines', name="σ² = " + str(sigma4 ** 2), line=dict(color='#8FD9FB')),
        row=1, col=4)
    fig.add_vline(x=x_min4, line=dict(color='#8FD9FB', dash='dash'), annotation_text="s*",
                  annotation=dict(font=dict(color='#8FD9FB')), annotation_position="top", row=1, col=4)
    # fig.update_yaxes(tickvals=[0, 0.5, 1], row=1, col=4)
    # fig.update_xaxes(tickvals=[-6, -3, 0, 3, 6], row=1, col=4)

    fig.update_layout(
        width=1500, height=300,
        template="plotly_white",
        font=dict(size=20)
    )
    fig.update_annotations(font_size=20)
    fig.update_layout(showlegend=False)

    fig.write_image(f"../results/plots_Example2/pdf_2_gaussian.png")


def plot_cdf_2_gaussian(mu1, mu2, eps1, eps2, sigma1, sigma2, sigma3, sigma4):
    """
    Create a figure with 4 subfigures: the ecdf of the distribution for 4 values of sigma.
    """

    x = np.linspace(-5, 4.5, 1000)

    # Cdf
    mixture_cdf1 = eps1 * norm.cdf(x, mu1, sigma1) + eps2 * norm.cdf(x, mu2, sigma1)
    mixture_cdf2 = eps1 * norm.cdf(x, mu1, sigma2) + eps2 * norm.cdf(x, mu2, sigma2)
    mixture_cdf3 = eps1 * norm.cdf(x, mu1, sigma3) + eps2 * norm.cdf(x, mu2, sigma3)
    mixture_cdf4 = eps1 * norm.cdf(x, mu1, sigma4) + eps2 * norm.cdf(x, mu2, sigma4)

    fig = make_subplots(rows=1, cols=4, start_cell="top-left",  x_title="x", y_title="cdf(x)")

    fig.add_trace(go.Scatter(x=x, y=mixture_cdf1, mode='lines', name="σ² = "+str(sigma1**2), line=dict(color='darkblue')),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=mixture_cdf2, mode='lines', name="σ² = "+str(sigma2**2), line=dict(color='#2D68C4')),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=x, y=mixture_cdf3, mode='lines', name="σ² = "+str(sigma3**2), line=dict(color='#2f97ff')),
                  row=1, col=3)

    fig.add_trace(go.Scatter(x=x, y=mixture_cdf4, mode='lines', name="σ² = "+str(sigma4**2), line=dict(color='#8FD9FB')),
                  row=1, col=4)
    # fig.update_yaxes(tickvals=[-0.5, 0, 0.5, 1, 1.5], row=1, col=4)
    # fig.update_xaxes(tickvals=[-9, -6, -3, 0, 3, 6, 9], row=1, col=4)

    fig.update_layout(
        width=1500, height=300,
        template="plotly_white",
        font=dict(size=20)
    )
    fig.update_annotations(font_size=20)
    fig.update_layout(showlegend=False)

    fig.write_image(f"../results/plots_Example2/cdf_2_gaussian.png")


if __name__ == "__main__":

    # Plot 1: Phi* and s* as a function of sigma^2
    plot_ftu_and_pivot_vs_sigma_sq()

    # Plot 2: Density of Gaussian mixture 1x5
    plot_pdf_2_gaussian(mu1, mu2, eps1, eps2, sigma1, sigma2, sigma3, sigma4)

    # Plot 3: Cdf of Gaussian mixture 1x5
    plot_cdf_2_gaussian(mu1, mu2, eps1, eps2, sigma1, sigma2, sigma3, sigma4)

