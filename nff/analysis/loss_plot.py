import matplotlib.pyplot as plt
import numpy as np

from . import mpl_settings


def plot_loss(energy_history, forces_history, figname, train_key='train', val_key='val'):
    """ Plot the loss history of the model.
    Args:
        energy_history (dict): energy loss history of the model for training and validation
        forces_history (dict): forces loss history of the model for training and validation
        figname (str): name of the figure

    Returns:
        None
    """
    epochs = np.arange(1, len(energy_history[train_key]) + 1)
    fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6), dpi=mpl_settings.DPI)
    ax_fig[0].semilogy(epochs, energy_history[train_key], label='train', color=mpl_settings.colors[1])
    ax_fig[0].semilogy(epochs, energy_history[val_key], label='val', color=mpl_settings.colors[2])
    ax_fig[0].legend()
    ax_fig[0].set_xlabel('Epoch')
    ax_fig[0].set_ylabel('Loss')

    ax_fig[1].semilogy(epochs, forces_history[train_key], label='train', color=mpl_settings.colors[1])
    ax_fig[1].semilogy(epochs, forces_history[val_key], label='val', color=mpl_settings.colors[2])
    ax_fig[1].legend()
    ax_fig[1].set_xlabel('Epoch')
    ax_fig[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()
