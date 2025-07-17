import matplotlib.pyplot as plt
import numpy as np

from . import mpl_settings


def plot_loss(
    energy_history: dict,
    forces_history: dict,
    figname: str,
    train_key: str = "train",
    val_key: str = "val",
) -> None:
    """Plot the loss history of the model.

    Args:
        energy_history: energy loss history of the model for training and validation
        forces_history: forces loss history of the model for training and validation
        figname: name of the figure
        train_key: key for training data in the history dictionary
        val_key: key for validation data in the history dictionary
    """
    epochs = np.arange(1, len(energy_history[train_key]) + 1)
    fig, ax_fig = plt.subplots(1, 2, figsize=(5, 2.5), dpi=mpl_settings.DPI)
    ax_fig[0].semilogy(epochs, energy_history[train_key], label="train", color=mpl_settings.colors[1])
    ax_fig[0].semilogy(epochs, energy_history[val_key], label="val", color=mpl_settings.colors[2])
    ax_fig[0].legend()
    ax_fig[0].set_xlabel("Epoch")
    ax_fig[0].set_ylabel("Loss")

    ax_fig[1].semilogy(epochs, forces_history[train_key], label="train", color=mpl_settings.colors[1])
    ax_fig[1].semilogy(epochs, forces_history[val_key], label="val", color=mpl_settings.colors[2])
    ax_fig[1].legend()
    ax_fig[1].set_xlabel("Epoch")
    ax_fig[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()
