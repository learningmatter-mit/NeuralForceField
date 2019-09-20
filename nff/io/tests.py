from .ase import *
import networkx as nx

def get_ethanol():
    """Returns an ethanol molecule.

    Returns:
        ethanol (Atoms)
    """
    nxyz = np.array([
	[ 6.0000e+00,  5.5206e-03,  5.9149e-01, -8.1382e-04],
        [ 6.0000e+00, -1.2536e+00, -2.5536e-01, -2.9801e-02],
        [ 8.0000e+00,  1.0878e+00, -3.0755e-01,  4.8230e-02],
        [ 1.0000e+00,  6.2821e-02,  1.2838e+00, -8.4279e-01],
        [ 1.0000e+00,  6.0567e-03,  1.2303e+00,  8.8535e-01],
        [ 1.0000e+00, -2.2182e+00,  1.8981e-01, -5.8160e-02],
        [ 1.0000e+00, -9.1097e-01, -1.0539e+00, -7.8160e-01],
        [ 1.0000e+00, -1.1920e+00, -7.4248e-01,  9.2197e-01],
        [ 1.0000e+00,  1.8488e+00, -2.8632e-02, -5.2569e-01]
    ])
    ethanol = Atoms(
        nxyz[:, 0].astype(int),
        positions=nxyz[:, 1:]
    )

    return ethanol

def test_AtomsBatch():
    # Test for an ethanol molecule (no PBC)
    ethanol = get_ethanol()

    expected_nbrlist_cutoff_2dot5 = np.array([ 
	[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
        [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 0],
        [2, 1], [2, 3], [2, 4], [2, 6], [2, 7], [2, 8], [3, 0], [3, 1],
        [3, 2], [3, 4], [3, 8], [4, 0], [4, 1], [4, 2], [4, 3], [4, 7],
        [5, 0], [5, 1], [5, 6], [5, 7], [6, 0], [6, 1], [6, 2], [6, 5],
        [6, 7], [7, 0], [7, 1], [7, 2], [7, 4], [7, 5], [7, 6], [8, 0],
        [8, 2], [8, 3]
    ])

    atoms_batch = AtomsBatch(ethanol, cutoff=2.5)
    atoms_batch.update_nbr_list()

    G1 = nx.from_edgelist(expected_nbrlist_cutoff_2dot5)
    G2 = nx.from_edgelist(atoms_batch.nbr_list.numpy())
    
    assert nx.is_isomorphic(G1, G2)




