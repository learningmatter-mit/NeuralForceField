import torch
import numpy as np


def get_neighbor_list(xyz, cutoff=5, undirected=True):
    """Get neighbor list from xyz positions of atoms.

    Args:
        xyz (torch.Tensor or np.array): (N, 3) array with positions
            of the atoms.
        cutoff (float): maximum distance to consider atoms as
            connected.

    Returns:
        nbr_list (torch.Tensor): (num_edges, 2) array with the
            indices of connected atoms.
    """

    xyz = torch.Tensor(xyz)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = mask.nonzero()

    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list

    