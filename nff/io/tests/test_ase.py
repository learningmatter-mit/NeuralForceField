from nff.io.ase import *
import numpy as np
import unittest as ut
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


@ut.skip('skip this for now')
class TestAtomsBatch(ut.TestCase):
    def setUp(self):
        self.ethanol = get_ethanol()

    def test_AtomsBatch():
        # Test for an ethanol molecule (no PBC)

        expected_nbrlist_cutoff_2dot5 = np.array([ 
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
            [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 0],
            [2, 1], [2, 3], [2, 4], [2, 6], [2, 7], [2, 8], [3, 0], [3, 1],
            [3, 2], [3, 4], [3, 8], [4, 0], [4, 1], [4, 2], [4, 3], [4, 7],
            [5, 0], [5, 1], [5, 6], [5, 7], [6, 0], [6, 1], [6, 2], [6, 5],
            [6, 7], [7, 0], [7, 1], [7, 2], [7, 4], [7, 5], [7, 6], [8, 0],
            [8, 2], [8, 3]
        ])

        atoms_batch = AtomsBatch(self.ethanol, cutoff=2.5)
        atoms_batch.update_nbr_list()

        G1 = nx.from_edgelist(expected_nbrlist_cutoff_2dot5)
        G2 = nx.from_edgelist(atoms_batch.nbr_list.numpy())
        
        assert nx.is_isomorphic(G1, G2)

    def test_get_batch():
        atoms_batch = AtomsBatch(self.ethanol, cutoff=5)
        batch = atoms_batch.get_batch()

        assert 'nxyz' in batch


class TestPeriodic(ut.TestCase):
    def setUp(self):
        nxyz = np.array([
            [14.0, -1.19984241582007, 2.07818802527655, 4.59909615202747],
            [14.0, 1.31404847917993, 2.27599872954824, 2.7594569553608],
            [14.0, 2.39968483164015, 0.0, 0.919817758694137],
            [8.0, -1.06646793438585, 3.24694318819338, 0.20609293956337],
            [8.0, 0.235189576572621, 1.80712683722845, 3.8853713328967],
            [8.0, 0.831278357813231, 3.65430348422777, 2.04573213623004],
            [8.0, 3.34516925281323, 0.699883270597028, 5.31282465043663],
            [8.0, 1.44742296061415, 1.10724356663142, 1.6335462571033],
            [8.0, 2.74908047157262, 2.54705991759635, 3.47318545376996]
        ])
        lattice = np.array([
            [5.02778179, 0.0, 3.07862843796742e-16],
            [-2.513890895, 4.3541867548248, 3.07862843796742e-16],
            [0.0, 0.0, 5.51891759]
        ])
        self.quartz = AtomsBatch(
            nxyz[:, 0].astype(int),
            positions=nxyz[:, 1:],
            cell=lattice,
            pbc=True
        )

    def test_ase(self):
        print(self.quartz)

    def test_nbrlist(self):
        nbrlist, offsets = self.quartz.update_nbr_list()
        print(nbrlist)
        print(offsets)



if __name__ == '__main__':
    ut.main()
