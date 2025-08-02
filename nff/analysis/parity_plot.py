from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from torch import Tensor
from nff.data import to_tensor
from nff.utils import cuda

from . import mpl_settings

plt.style.use("ggplot")
mpl_settings.update_custom_settings()


def plot_parity(
    results: Dict[str, list | Tensor],
    targets: Dict[str, list | Tensor],
    figname: str,
    plot_type: Literal["hexbin", "scatter"] = "hexbin",
    energy_key: str = "energy",
    force_key: str = "energy_grad",
    units: Dict[str, str] = {"energy": "eV", "energy_grad": "eV/Ang"},
) -> tuple[float, float]:
    """Perform a parity plot between the results and the targets.

    Args:
        results: dictionary containing the results
        targets: dictionary containing the targets
        figname: name of the figure
        plot_type: type of plot to use, either "hexbin" or "scatter"
        energy_key: key for the energy
        force_key: key for the forces
        units: dictionary containing the units of the keys

    Returns:
        float: MAE of the energy
        float: MAE of the forces
    """

    fig, ax_fig = plt.subplots(1, 2, figsize=(5, 2.5), dpi=mpl_settings.DPI)

    mae_save = {force_key: 0, energy_key: 0}

    results = cuda.batch_detach(results)
    targets = cuda.batch_detach(targets)

    for ax, key in zip(ax_fig, units.keys()):
        pred = to_tensor(results[key], stack=True).numpy()
        targ = to_tensor(targets[key], stack=True).numpy()

        mae = abs(pred - targ).mean()
        mae_save[key] = mae

        lim_min = min(np.min(pred), np.min(targ))
        lim_max = max(np.max(pred), np.max(targ))

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
                rasterized=True,
            )

        else:
            hb = ax.scatter(pred, targ, color="#ff7f0e", alpha=0.3, rasterized=True)

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Counts")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        ax.plot((lim_min, lim_max), (lim_min, lim_max), color="#000000", zorder=-1)

        label = key
        ax.set_title(label.upper())
        ax.set_xlabel(f"Predicted {label} [{units[key]}]")
        ax.set_ylabel(f"Target {label} [{units[key]}]")

        ax.text(
            0.1,
            0.9,
            f"MAE: {mae:.2f} {units[key]}",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(f"{figname}.pdf")
    plt.show()
    mae_energy = float(mae_save[energy_key])
    mae_forces = float(mae_save[force_key])
    return mae_energy, mae_forces


def plot_err_var(
    err: Tensor | np.ndarray,
    var: Tensor | np.ndarray,
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
        err: error of the forces
        var: variance of the forces
        figname: name of the figure
        units: units of the error and variance
        x_min: minimum value of the x-axis
        x_max: maximum value of the x-axis
        y_min: minimum value of the y-axis
        y_max: maximum value of the y-axis
        sample_frac: fraction of the data to sample for the plot
        num_bins: number of bins to use for binning
        cb_format: format of the colorbar
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
    ax.text(
        0.6,
        0.9,
        rf"$R^2$: {res.rvalue**2:.3f}",
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
