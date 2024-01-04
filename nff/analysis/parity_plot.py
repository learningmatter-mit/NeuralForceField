import matplotlib.pyplot as plt
import numpy as np
import torch
from nff.data.dataset import to_tensor
from nff.utils import constants, cuda

from . import mpl_settings


def plot_parity(results, targets, figname, plot_type="hexbin", energy_key="energy", force_key="energy_grad", units={"energy": "eV", "energy_grad": "eV/Ang"}):
    fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6), dpi=mpl_settings.DPI)

    # if convert_ev:
    #     units = {force_key: r"eV/$\AA$", energy_key: "eV"}
    # else:
    #     units = {force_key: r"kcal/mol/$\AA$", energy_key: "kcal/mol"}

    mae_save = {force_key: 0, energy_key: 0}

    results = cuda.batch_detach(results)
    targets = cuda.batch_detach(targets)

    for ax, key in zip(ax_fig, units.keys()):
        pred = to_tensor(results[key], stack=True)
        targ = to_tensor(targets[key], stack=True)

        # if convert_ev:
        #     pred = pred * 1/constants.EV_TO_KCAL_MOL
        #     targ = targ * 1/constants.EV_TO_KCAL_MOL

        # # per atom energy MAE
        # if per_atom and key in energy_key:
        #     num_atoms = torch.Tensor(
        #         [num_atoms.item() for num_atoms in targets["num_atoms"]]
        #     )
        #     pred = pred / num_atoms
        #     targ = targ / num_atoms
        #     units[energy_key] += "/atom"
        #     mae = abs(pred - targ).mean() * 1000  # convert eV to meV
        # else:
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

        # label = key.replace(force_key, "force")
        label = key
        ax.set_title(label.upper())
        ax.set_xlabel("Predicted %s [%s]" % (label, units[key]))
        ax.set_ylabel("Target %s [%s]" % (label, units[key]))
        # if per_atom and key in energy_key:
        #     ax.text(
        #         0.1,
        #         0.9,
        #         "MAE: %.2f %s" % (mae, units[key].replace("eV", "meV")),
        #         transform=ax.transAxes,
        #     )
        # else:
        ax.text(
            0.1,
            0.9,
            "MAE: %.2f %s" % (mae, units[key]),
            transform=ax.transAxes,
        )

        # increase tick size and make them point inwards
        # ax.tick_params(
        #     axis="y",
        #     direction="in",
        # )
        # ax.tick_params(
        #     axis="x",
        #     direction="in",
        # )
        # cb.ax.tick_params(
        #     axis="both",
        #     which="major",
       
        # )
        # cb.ax.tick_params(
        #     axis="both",
        #     which="minor",
        # )

        # np.savetxt(key, torch.stack((pred.view(-1), targ.view(-1)), dim=1))

    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()
    mae_energy = float(mae_save[energy_key])
    mae_forces = float(mae_save[force_key])
    return mae_energy, mae_forces