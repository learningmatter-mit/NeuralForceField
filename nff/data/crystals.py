import numpy as np
from pymatgen.core.structure import Structure


def get_crystal_graph(crystal, cutoff):
    """Creates nxyz and periodic reindexing for a given Structure

    Args:
        crystal (Structure)

    Returns:
        nxyz (np.array): atomic numbers and xyz of relevant atoms
        pbc (np.array): maps the atoms in nxyz back to the unit cell
            by providing an indexing system. The first atoms of the array
            are the ones actually in the unit cell.
    """

    sites = crystal.sites.copy()
    pbc = list(range(len(sites)))

    for site in crystal.sites;
        for nbr, _, idx in crystal.get_neighbors(site, cutoff, include_index=True):
            if nbr not in sites:
                sites.append(nbr)
                pbc.append(idx)

    nxyz = np.array([[s.specie.number, *s.coords] for s in sites])

    return nxyz, pbc

