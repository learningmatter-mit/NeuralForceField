import torch
import numpy as np
import pdb
from scipy.spatial.distance import cdist

DISTANCETHRESHOLDICT = {
    (1., 1.): 1.00,
    (1., 3.): 1.30,
    (1., 5.): 1.50,
    (1., 6.): 1.30,
    (1., 7.): 1.30,
    (1., 8.): 1.30,
    (1., 9.): 1.30,
    (1., 11.): 1.65,
    (1., 14.): 1.65,
    (1., 16.): 1.50,
    (1., 17.): 1.60,
    (1., 35.): 1.60,
    (3., 6.): 1.60,
    (3., 7.): 1.60,
    (3., 8.): 1.60,
    (3., 9.): 1.60,
    (5., 11.): 1.8,
    (5., 17.): 2.1,
    (5., 35.): 2.1,
    (5., 6.): 1.70,
    (5., 7.): 1.70,
    (5., 8.): 1.70,
    (5., 9.): 1.70,
    (5., 11.): 1.70,
    (6., 6.): 1.70,
    (6., 8.): 1.62,
    (6., 9.): 1.65,
    (6., 11.): 1.70,
    (6., 14.): 2.10,
    (6., 16.): 2.20,
    (7., 8.): 1.55,
    (7., 11.): 1.70,
    (8., 11.): 1.70,
    (8., 16.): 2.10,
    (8., 17.): 1.80,
    (8., 8.): 1.70,
    (8., 9.): 1.50,
    (8., 14.): 1.85,
    (8., 35.): 1.70, }

DISTANCETHRESHOLD = 2.0

def get_neighbor_list(xyz, cutoff=5):
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

    return nbr_list


def distancemat(nxyz):

    """
    Get the NxN distance matrix for a set of atoms (N is the number of atoms).
    Args:
        nxyz (list): atomic number and coordinates in the form [[Z1, x1, y1, z1],
            [Z2, x2, y2, z2], ...]
    Returns:
        atomicnums (list): list of atomic numbers
        cdist(xyz, xyz) (list): distance matrix

    """
    xyz = np.array(nxyz)[:, 1:]
    atomicnums = np.array(nxyz)[:, 0].astype(int)
    return atomicnums, cdist(xyz, xyz)

def adjdistmat(nxyz, threshold=DISTANCETHRESHOLD):

    """ 
    Get the adjacency matrix and the distance matrix from nxyz.
    Args:
        nxyz (list): atomic number and coordinates in the form [[Z1, x1, y1, z1],
            [Z2, x2, y2, z2], ...]
        threshold: default threshold for two atoms to be considered bonded if
            the types of atoms in question are not included in DISTANCETHRESHOLDICT

    Returns:
        np.array(atomicnums) (numpy.ndarray): atomic numbers
        np.array(adjmat) (numpy.ndarray): adjacency matrix. This is an NxN matrix
            with zeros everywhere but 1 for indices (i, j) if i and j are bonded.
        np.array(dmat): NxN matrix with entries r_ij, the distance between atoms
            i and j.

    Example: CH4.
        nxyz = [[6.0, 0.0, 0.0, 0.0],
            [1.0, -0.5985, 0.2627, 0.875],
            [1.0, -0.5209, -0.7639, -0.5814],
            [1.0, 0.1508, 0.888, -0.6178],
            [1.0, 0.9686, -0.3868, 0.3242]]
        adjistmat(nxyz)
            >> (array([6, 1, 1, 1, 1]),
                array([[0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0]]),
                array([[0.        , 1.09217148, 1.09220235, 1.09222776, 1.09220229],
                    [1.09217148, 0.        , 1.7835443 , 1.78350846, 1.78354515],
                    [1.09220235, 1.7835443 , 0.        , 1.78361416, 1.78351507],
                    [1.09222776, 1.78350846, 1.78361416, 0.        , 1.78361315],
                    [1.09220229, 1.78354515, 1.78351507, 1.78361315, 0.        ]])
                )

    """

    atomicnums, dmat = distancemat(nxyz)
    thresholdmat = np.array([[threshold.get(tuple(
        sorted((i, j))), DISTANCETHRESHOLD) for i in atomicnums] for j in atomicnums])
    adjmat = (dmat < thresholdmat).astype(int)
    np.fill_diagonal(adjmat, 0)
    return np.array(atomicnums), np.array(adjmat), np.array(dmat)

def get_bond_list(nxyz):

    """
    Get a list of bonded pairs.
    Args:
        nxyz (list): atomic number and coordinates in the form [[Z1, x1, y1, z1],
            [Z2, x2, y2, z2], ...]
    Returns:
        mol_bond.tolist() (list): list of bonded pairs
    Example: CH4.
        nxyz = [[6.0, 0.0, 0.0, 0.0],
            [1.0, -0.5985, 0.2627, 0.875],
            [1.0, -0.5209, -0.7639, -0.5814],
            [1.0, 0.1508, 0.888, -0.6178],
            [1.0, 0.9686, -0.3868, 0.3242]]
        get_bond_list(nxyz)
        >> [[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0], [4, 0]]

    """

    atomicnums, adjmat, dmat = adjdistmat(nxyz=nxyz, threshold=DISTANCETHRESHOLDICT)
    mol_bond = np.array(adjmat.nonzero()).transpose()
    
    return mol_bond.tolist()
