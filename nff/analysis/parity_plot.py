import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nff.data.dataset import to_tensor
from nff.utils import cuda

from . import mpl_settings


def plot_parity(results, targets, figname, plot_type="hexbin", energy_key="energy", force_key="energy_grad", units={"energy": "eV", "energy_grad": "eV/Ang"}):
    fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6), dpi=mpl_settings.DPI)

    mae_save = {force_key: 0, energy_key: 0}

    results = cuda.batch_detach(results)
    targets = cuda.batch_detach(targets)

    for ax, key in zip(ax_fig, units.keys()):
        pred = to_tensor(results[key], stack=True)
        targ = to_tensor(targets[key], stack=True)

        mae = abs(pred - targ).mean()
        mae_save[key] = mae

        if plot_type.lower() == "hexbin":
            hb = ax.hexbin(
                pred,
                targ,
                mincnt=1,
                gridsize=(mpl_settings.GRIDSIZE, mpl_settings.GRIDSIZE),
                bins="log",
                cmap=mpl_settings.cmap,
                edgecolor="None",
            )

        else:
            hb = ax.scatter(pred, targ, color="#ff7f0e", alpha=0.3)

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Counts")

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

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        ax.plot(
            (lim_min, lim_max),
            (lim_min, lim_max),
            color="#000000",
            zorder=-1
        )

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

def plot_err_var(err, var, figname, units="eV/Å", x_min=0, x_max=1, y_min=0, y_max=1, sample_frac=1.0, num_bins=10, cb_format="%.2f"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=mpl_settings.DPI)

    idx = np.arange(len(var))
    np.random.seed(2)
    sample_idx = np.random.choice(idx, size=int(len(idx) * sample_frac), replace=False)
    n_samples = len(sample_idx)

    var = var.flatten()[sample_idx]
    err = err.flatten()[sample_idx]

    # binning force var and force MAE
    from scipy import stats

    err_binned, var_binned_edges, bin_nums = stats.binned_statistic(
        var, err, statistic="mean", bins=num_bins, range=(x_min, x_max)
    )
    bin_width = var_binned_edges[1] - var_binned_edges[0]
    var_binned_centers = var_binned_edges[1:] - bin_width / 2
    res = stats.linregress(var_binned_centers, err_binned)

    # plot density kernel
    x = pd.Series(var)
    y = pd.Series(err)
    from scipy.stats import gaussian_kde

    kernel = gaussian_kde(
        np.vstack(
            [x.sample(n=len(x), random_state=2), y.sample(n=len(y), random_state=2)]
        )
    )
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

    from matplotlib.lines import Line2D

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