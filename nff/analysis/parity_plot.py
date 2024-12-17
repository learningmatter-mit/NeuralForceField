from typing import Dict, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

from nff.data import to_tensor
from nff.utils import cuda

from . import mpl_settings


def plot_parity(
    results: Dict[str, Union[list, torch.Tensor]],
    targets: Dict[str, Union[list, torch.Tensor]],
    figname: str,
    plot_type: Literal["hexbin", "scatter"] = "hexbin",
    energy_key: str = "energy",
    force_key: str = "energy_grad",
    units: Dict[str, str] = {"energy": "eV", "energy_grad": "eV/Ang"},
) -> tuple[float, float]:
    """
    Perform a parity plot between the results and the targets.

    Args:
        results (dict): dictionary containing the results
        targets (dict): dictionary containing the targets
        figname (str): name of the figure
        plot_type (str): type of plot to use, either "hexbin" or "scatter"
        energy_key (str): key for the energy
        force_key (str): key for the forces
        units (dict): dictionary containing the units of the keys

    Returns:
        float: MAE of the energy
        float: MAE of the forces
    """

    fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6), dpi=mpl_settings.DPI)

    mae_save = {force_key: 0, energy_key: 0}

    results = cuda.batch_detach(results)
    targets = cuda.batch_detach(targets)

    for ax, key in zip(ax_fig, units.keys()):
        pred = to_tensor(results[key], stack=True)
        targ = to_tensor(targets[key], stack=True)

        mae = abs(pred - targ).mean()
        mae_save[key] = mae

        lim_min = min(torch.min(pred), torch.min(targ))
        lim_max = max(torch.max(pred), torch.max(targ))

        if lim_min < 0:
            lim_min *= 1.1
        else:
            lim_min *= 0.9

        if lim_max < 0:
            lim_max *= 0.9
        else:
            lim_max *= 1.1

        if plot_type.lower() == "hexbin":
            hb = ax.hexbin(
                pred,
                targ,
                mincnt=1,
                gridsize=(mpl_settings.GRIDSIZE, mpl_settings.GRIDSIZE),
                bins="log",
                cmap=mpl_settings.cmap,
                edgecolor="None",
                extent=(lim_min, lim_max, lim_min, lim_max),
            )

        else:
            hb = ax.scatter(pred, targ, color="#ff7f0e", alpha=0.3)

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Counts")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        ax.plot((lim_min, lim_max), (lim_min, lim_max), color="#000000", zorder=-1)

        label = key
        ax.set_title(label.upper())
        ax.set_xlabel("Predicted %s [%s]" % (label, units[key]))
        ax.set_ylabel("Target %s [%s]" % (label, units[key]))

        ax.text(
            0.1,
            0.9,
            "MAE: %.2f %s" % (mae, units[key]),
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()
    mae_energy = float(mae_save[energy_key])
    mae_forces = float(mae_save[force_key])
    return mae_energy, mae_forces


def plot_err_var(
    err: Union[torch.Tensor, np.ndarray],
    var: Union[torch.Tensor, np.ndarray],
    figname: str,
    units: str = "eV/Å",
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    sample_frac: float = 1.0,
    num_bins: int = 10,
    cb_format: str = "%.2f",
) -> None:
    """Plot the error vs variance of the forces.

    Args:
        err (torch.Tensor): error of the forces
        var (torch.Tensor): variance of the forces
        figname (str): name of the figure
        units (str): units of the error and variance
        x_min (float): minimum value of the x-axis
        x_max (float): maximum value of the x-axis
        y_min (float): minimum value of the y-axis
        y_max (float): maximum value of the y-axis
        sample_frac (float): fraction of the data to sample for the plot
        num_bins (int): number of bins to use for binning
        cb_format (str): format of the colorbar

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=mpl_settings.DPI)

    idx = np.arange(len(var))
    np.random.seed(2)
    sample_idx = np.random.choice(idx, size=int(len(idx) * sample_frac), replace=False)
    len(sample_idx)

    var = var.flatten()[sample_idx]
    err = err.flatten()[sample_idx]

    # binning force var and force MAE

    err_binned, var_binned_edges, bin_nums = stats.binned_statistic(
        var, err, statistic="mean", bins=num_bins, range=(x_min, x_max)
    )
    bin_width = var_binned_edges[1] - var_binned_edges[0]
    var_binned_centers = var_binned_edges[1:] - bin_width / 2
    res = stats.linregress(var_binned_centers, err_binned)

    # plot density kernel
    x = pd.Series(var)
    y = pd.Series(err)

    kernel = gaussian_kde(np.vstack([x.sample(n=len(x), random_state=2), y.sample(n=len(y), random_state=2)]))
    c = kernel(np.vstack([x, y]))
    hb = ax.scatter(
        var,
        err,
        c=c,
        cmap=mpl_settings.cmap,
        edgecolors="None",
        label="Sampled points",
        rasterized=True,
    )
    avg = ax.scatter(
        var_binned_centers,
        err_binned,
        marker="X",
        edgecolors="k",
        label="Binned avg.",
        zorder=2,
    )

    (bf_line,) = ax.plot(
        x.sample(n=len(x), random_state=2),
        res.intercept + res.slope * x.sample(n=len(x), random_state=2),
        c="k",
        ls="-",
        label="Avg. best fit",
        zorder=1,
    )
    min_text = ax.text(
        0.6,
        0.9,
        r"$R^2$: {:.3f}".format(res.rvalue**2),
        transform=ax.transAxes,
    )

    cb = fig.colorbar(hb, ax=ax, format=cb_format)
    cb.set_label("Estimated probability density")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(f"Force SD [{units}]", labelpad=5)
    ax.set_ylabel(f"Force MAE [{units}]", labelpad=5)

    scatter_leg = (
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            markeredgewidth=0,
            markeredgecolor=None,
            markerfacecolor="k",
            # alpha=ALPHA,
        ),
    )

    labels = ["Sampled point", "Binned avg.", "Avg. best fit"]
    handles = (scatter_leg, avg, bf_line)

    leg = ax.legend(handles, labels, loc="lower right", frameon=True)
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_boxstyle("Square", pad=0)

    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()
