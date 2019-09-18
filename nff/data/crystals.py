import torch
import numpy as np
from pymatgen.core.structure import Structure


def get_crystal_graph(crystal, cutoff):
    """Creates nxyz and periodic reindexing for a given Structure.

    Args:
        crystal (Structure)
        cutoff (float): cutoff to get neighbors of atoms

    Returns:
        nxyz (np.array): atomic numbers and xyz of relevant atoms
        pbc (np.array): maps the atoms in nxyz back to the unit cell
            by providing an indexing system. The first atoms of the array
            are the ones actually in the unit cell.
    """

    sites = crystal.sites.copy()
    pbc = list(range(len(sites)))

    for site in crystal.sites;
        for nbr in crystal.get_neighbors(site, cutoff):
            if nbr.site not in sites:
                sites.append(nbr.site)
                pbc.append(nbr.index)

    nxyz = torch.Tensor([[s.specie.number, *s.coords] for s in sites])
    pbc = torch.LongTensor(pbc)

    return nxyz, pbc

