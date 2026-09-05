import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import generate_gaussian_mixture, generate_dirac_mixture


def generate_gauss_unif_mixture(x, eps, n, sigma=0.5):
    groups = []
    groups.append(np.random.normal(loc=x[0], scale=sigma, size=round(n*eps[0])))
    groups.append(np.random.uniform(low=x[1], high=x[2], size=round(n*eps[1])))
    data = np.concatenate(groups)
    return data


def plot_save_histo(data, i, binsize, x_range):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, histfunc="count", xbins=dict(size=binsize), marker_color="lightblue"))
    fig.update_xaxes(showticklabels=False, range=x_range)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(width=700, height=400,
                      margin=dict(l=5, r=5, t=5, b=5),
                      template="plotly_white")
    fig.write_image(f"../results/plots_tableSPL/histo{i}.png")


def plot_save_ecdf(data, i):
    fig = px.ecdf(data)
    fig.update_xaxes(showticklabels=False, showgrid=False, title="")
    fig.update_yaxes(showticklabels=False, title="")
    fig.update_traces(line=dict(width=8))
    fig.update_layout(width=700, height=400,
                      margin=dict(l=5, r=5, t=5, b=5),
                      template="plotly_white", showlegend=False)
    fig.write_image(f"../results/plots_tableSPL/ecdf{i}.png")


# 1: N(0, 1)
data = generate_gaussian_mixture(mu=np.array([0]),
                                 eps=np.array([1]),
                                 sigma=1, n=2000)
fig = plot_save_histo(data, i=1, binsize=0.2, x_range=[-3, 3])

plot_save_ecdf(data, 1)


# 2: 0.5 (N(0, 0.5) + N(1, 0.5))
data = generate_gaussian_mixture(mu=np.array([0, 1]),
                                 eps=np.array([0.5, 0.5]),
                                 sigma=0.5, n=2000)
fig = plot_save_histo(data, i=2, binsize=0.2, x_range=[-3, 3])
plot_save_ecdf(data, 2)


# 3: 0.6 N(0, 0.5) + 0.4 U(4, 8)
data = generate_gauss_unif_mixture(x=np.array([0, 4, 8]),
                                   eps=np.array([0.6, 0.4]),
                                   sigma=0.5, n=2000)
fig = plot_save_histo(data, i=3, binsize=0.36, x_range=[-3, 8])
plot_save_ecdf(data, 3)


# 4: 0.6 N(0, 0.5) + 0.4 U(1, 4)
data = generate_gauss_unif_mixture(x=np.array([0, 1, 4]),
                                   eps=np.array([0.6, 0.4]),
                                   sigma=0.5, n=2000)
fig = plot_save_histo(data, i=4, binsize=0.36, x_range=[-3, 8])
plot_save_ecdf(data, 4)


# 5: 0.2 D(-2) + 0.4 D(0) + 0.4 D(2)
data = generate_dirac_mixture(mu=np.array([-2, 0, 2]),
                              eps=np.array([0.2, 0.4, 0.4]),
                              n=2000)
fig = plot_save_histo(data, i=5, binsize=0.23, x_range=[-3.5, 3.5])
plot_save_ecdf(data, 5)


# 6: 0.2 N(-2, 0.5) + 0.4 N(0, 0.5) + 0.4 N(2, 0.5)
data = generate_gaussian_mixture(mu=np.array([-2, 0, 2]),
                                 eps=np.array([0.2, 0.4, 0.4]),
                                 sigma=0.5, n=2000)
fig = plot_save_histo(data, i=6, binsize=0.23, x_range=[-3.5, 3.5])
plot_save_ecdf(data, 6)

# 7: 0.2 (D(-3) + D(-1.5) + D(2.5) + D(4) + D(11))
data = generate_dirac_mixture(mu=np.array([-3, -1.5, 2.5, 4, 11]),
                              eps=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                              n=2000)
fig = plot_save_histo(data, i=7, binsize=0.56, x_range=[-4.5, 12.5])
plot_save_ecdf(data, 7)


# 8: 0.2 (N(-3, 0.5) + N(-1.5, 0.5) + N(2.5, 0.5) + N(4, 0.5) + N(11, 0.5))
data = generate_gaussian_mixture(mu=np.array([-3, -1.5, 2.5, 4, 11]),
                                 eps=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                                 sigma=0.5, n=2000)
fig = plot_save_histo(data, i=8, binsize=0.56, x_range=[-4.5, 12.5])
plot_save_ecdf(data, 8)

