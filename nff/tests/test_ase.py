import unittest as ut
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from nff.io.ase import AtomsBatch


def compare_dicts(d1: dict, d2: dict):
    """Compare the values of two dictionaries. Dictionaries are not nested.
    They can contain lists, numpy arrays, and scalars.

    Args:
        d1 (dict): The first dictionary.
        d2 (dict): The second dictionary.
    """
    for key, value in d1.items():
        if isinstance(value, dict):
            compare_dicts(value, d2[key])
        elif isinstance(value, str):
            assert value == d2[key]
        elif isinstance(value, Iterable):
            assert np.allclose(value, d2[key])
        else:
            assert value == d2[key]


def get_ethanol():
    """Returns an ethanol molecule.

    Returns:
        ethanol (Atoms)
    """
    nxyz = np.array(
        [
            [6.0000e00, 5.5206e-03, 5.9149e-01, -8.1382e-04],
            [6.0000e00, -1.2536e00, -2.5536e-01, -2.9801e-02],
            [8.0000e00, 1.0878e00, -3.0755e-01, 4.8230e-02],
            [1.0000e00, 6.2821e-02, 1.2838e00, -8.4279e-01],
            [1.0000e00, 6.0567e-03, 1.2303e00, 8.8535e-01],
            [1.0000e00, -2.2182e00, 1.8981e-01, -5.8160e-02],
            [1.0000e00, -9.1097e-01, -1.0539e00, -7.8160e-01],
            [1.0000e00, -1.1920e00, -7.4248e-01, 9.2197e-01],
            [1.0000e00, 1.8488e00, -2.8632e-02, -5.2569e-01],
        ]
    )
    return Atoms(nxyz[:, 0].astype(int), positions=nxyz[:, 1:])


@pytest.mark.usefixtures("device")  # Ensure the fixture is accessible
class TestAtomsBatch(ut.TestCase):
    def setUp(self):
        self.ethanol = get_ethanol()
        # Access the device value from the pytest fixture
        self.device = self._test_fixture_device

    @pytest.fixture(autouse=True)
    def inject_device(self, device):
        # Automatically set the fixture value to an attribute
        self._test_fixture_device = device

    @ut.skip("skip this for now")
    def test_AtomsBatch(self):
        # Test for an ethanol molecule (no PBC)

        expected_nbrlist_cutoff_2dot5 = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
                [1, 0],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [2, 0],
                [2, 1],
                [2, 3],
                [2, 4],
                [2, 6],
                [2, 7],
                [2, 8],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 8],
                [4, 0],
                [4, 1],
                [4, 2],
                [4, 3],
                [4, 7],
                [5, 0],
                [5, 1],
                [5, 6],
                [5, 7],
                [6, 0],
                [6, 1],
                [6, 2],
                [6, 5],
                [6, 7],
                [7, 0],
                [7, 1],
                [7, 2],
                [7, 4],
                [7, 5],
                [7, 6],
                [8, 0],
                [8, 2],
                [8, 3],
            ]
        )

        atoms_batch = AtomsBatch(self.ethanol, cutoff=2.5, device=self.device)
        atoms_batch.update_nbr_list()

        G1 = nx.from_edgelist(expected_nbrlist_cutoff_2dot5)
        G2 = nx.from_edgelist(atoms_batch.nbr_list.numpy())

        assert nx.is_isomorphic(G1, G2)

    def test_get_batch(self):
        atoms_batch = AtomsBatch(self.ethanol, cutoff=5, device=self.device)
        batch = atoms_batch.get_batch()

        assert "nxyz" in batch

    def test_from_atoms(self):
        atoms_batch = AtomsBatch.from_atoms(self.ethanol, cutoff=2.5, device=self.device)

        # ensure atomic numbers, positions, and cell are the same
        assert np.allclose(atoms_batch.get_atomic_numbers(), self.ethanol.get_atomic_numbers())
        assert np.allclose(atoms_batch.get_positions(), self.ethanol.get_positions())
        assert np.allclose(atoms_batch.get_cell(), self.ethanol.get_cell())

    def test_copy(self):
        atoms_batch = AtomsBatch(self.ethanol, cutoff=2.5, device=self.device)
        atoms_batch.get_batch()  # update props
        atoms_batch_copy = atoms_batch.copy()

        # ensure ASE attributes are the same
        assert np.allclose(atoms_batch.get_atomic_numbers(), atoms_batch_copy.get_atomic_numbers())
        assert np.allclose(atoms_batch.get_positions(), atoms_batch_copy.get_positions())
        assert np.allclose(atoms_batch.get_cell(), atoms_batch_copy.get_cell())

        # ensure NFF attributes are the same
        compare_dicts(atoms_batch.props, atoms_batch_copy.props)
        assert np.allclose(atoms_batch.nbr_list.numpy(), atoms_batch_copy.nbr_list.numpy())
        assert np.allclose(atoms_batch.offsets.numpy(), atoms_batch_copy.offsets.numpy())
        assert atoms_batch.directed == atoms_batch_copy.directed
        assert atoms_batch.cutoff == atoms_batch_copy.cutoff
        assert atoms_batch.cutoff_skin == atoms_batch_copy.cutoff_skin
        assert atoms_batch.device == atoms_batch_copy.device
        assert atoms_batch.requires_large_offsets == atoms_batch_copy.requires_large_offsets

    def test_fromdict(self):
        atoms_batch = AtomsBatch(self.ethanol, cutoff=2.5, device=self.device)
        ab_dict = atoms_batch.todict(update_props=True)
        ab_from_dict = AtomsBatch.fromdict(ab_dict)

        # ensure ASE attributes are the same
        assert np.allclose(atoms_batch.get_atomic_numbers(), ab_from_dict.get_atomic_numbers())
        assert np.allclose(atoms_batch.get_positions(), ab_from_dict.get_positions())
        assert np.allclose(atoms_batch.get_cell(), ab_from_dict.get_cell())

        # ensure NFF attributes are the same
        compare_dicts(atoms_batch.props, ab_from_dict.props)
        assert np.allclose(atoms_batch.nbr_list.numpy(), ab_from_dict.nbr_list.numpy())
        assert np.allclose(atoms_batch.offsets.numpy(), ab_from_dict.offsets.numpy())
        assert atoms_batch.directed == ab_from_dict.directed
        assert atoms_batch.cutoff == ab_from_dict.cutoff
        assert atoms_batch.cutoff_skin == ab_from_dict.cutoff_skin
        assert atoms_batch.device == ab_from_dict.device
        assert atoms_batch.requires_large_offsets == ab_from_dict.requires_large_offsets

        # ensure dict from new AtomsBatch is the same as the original dict
        ab_dict_again = ab_from_dict.todict(update_props=True)
        ab_dict_props = ab_dict.pop("props")
        ab_dict_again_props = ab_dict_again.pop("props")
        # compare all keys except for the props
        compare_dicts(ab_dict, ab_dict_again)
        # compare the props
        compare_dicts(ab_dict_props, ab_dict_again_props)


@pytest.mark.usefixtures("device")  # Ensure the fixture is loaded
class TestPeriodic(ut.TestCase):
    def setUp(self):
        nxyz = np.array(
            [
                [14.0, -1.19984241582007, 2.07818802527655, 4.59909615202747],
                [14.0, 1.31404847917993, 2.27599872954824, 2.7594569553608],
                [14.0, 2.39968483164015, 0.0, 0.919817758694137],
                [8.0, -1.06646793438585, 3.24694318819338, 0.20609293956337],
                [8.0, 0.235189576572621, 1.80712683722845, 3.8853713328967],
                [8.0, 0.831278357813231, 3.65430348422777, 2.04573213623004],
                [8.0, 3.34516925281323, 0.699883270597028, 5.31282465043663],
                [8.0, 1.44742296061415, 1.10724356663142, 1.6335462571033],
                [8.0, 2.74908047157262, 2.54705991759635, 3.47318545376996],
            ]
        )
        lattice = np.array(
            [
                [5.02778179, 0.0, 3.07862843796742e-16],
                [-2.513890895, 4.3541867548248, 3.07862843796742e-16],
                [0.0, 0.0, 5.51891759],
            ]
        )
        self.quartz = AtomsBatch(
            nxyz[:, 0].astype(int), positions=nxyz[:, 1:], cell=lattice, pbc=True, device=self._test_fixture_device
        )

    @pytest.fixture(autouse=True)
    def inject_device(self, device):
        # Automatically set the fixture value to an attribute
        self._test_fixture_device = device

    def test_print(self):
        print(self.quartz)

    def test_nbrlist(self):
        nbrlist, offsets = self.quartz.update_nbr_list()
        expected_nbrlist = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 4],
                [0, 4],
                [0, 4],
                [0, 4],
                [0, 4],
                [0, 4],
                [0, 5],
                [0, 5],
                [0, 5],
                [0, 5],
                [0, 5],
                [0, 5],
                [0, 5],
                [0, 6],
                [0, 6],
                [0, 6],
                [0, 6],
                [0, 6],
                [0, 6],
                [0, 7],
                [0, 7],
                [0, 7],
                [0, 7],
                [0, 7],
                [0, 7],
                [0, 7],
                [0, 8],
                [0, 8],
                [0, 8],
                [0, 8],
                [0, 8],
                [0, 8],
                [0, 8],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 4],
                [1, 4],
                [1, 4],
                [1, 4],
                [1, 4],
                [1, 4],
                [1, 4],
                [1, 5],
                [1, 5],
                [1, 5],
                [1, 5],
                [1, 5],
                [1, 5],
                [1, 6],
                [1, 6],
                [1, 6],
                [1, 6],
                [1, 6],
                [1, 6],
                [1, 6],
                [1, 7],
                [1, 7],
                [1, 7],
                [1, 7],
                [1, 7],
                [1, 7],
                [1, 7],
                [1, 8],
                [1, 8],
                [1, 8],
                [1, 8],
                [1, 8],
                [1, 8],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 4],
                [2, 4],
                [2, 4],
                [2, 4],
                [2, 4],
                [2, 4],
                [2, 4],
                [2, 5],
                [2, 5],
                [2, 5],
                [2, 5],
                [2, 5],
                [2, 5],
                [2, 5],
                [2, 6],
                [2, 6],
                [2, 6],
                [2, 6],
                [2, 6],
                [2, 6],
                [2, 6],
                [2, 7],
                [2, 7],
                [2, 7],
                [2, 7],
                [2, 7],
                [2, 7],
                [2, 8],
                [2, 8],
                [2, 8],
                [2, 8],
                [2, 8],
                [2, 8],
                [2, 8],
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 4],
                [3, 5],
                [3, 5],
                [3, 5],
                [3, 5],
                [3, 5],
                [3, 5],
                [3, 5],
                [3, 6],
                [3, 6],
                [3, 6],
                [3, 6],
                [3, 6],
                [3, 6],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 7],
                [3, 8],
                [3, 8],
                [3, 8],
                [3, 8],
                [3, 8],
                [3, 8],
                [3, 8],
                [3, 8],
                [4, 5],
                [4, 5],
                [4, 5],
                [4, 5],
                [4, 5],
                [4, 5],
                [4, 5],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 6],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 7],
                [4, 8],
                [4, 8],
                [4, 8],
                [4, 8],
                [4, 8],
                [4, 8],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 6],
                [5, 7],
                [5, 7],
                [5, 7],
                [5, 7],
                [5, 7],
                [5, 7],
                [5, 8],
                [5, 8],
                [5, 8],
                [5, 8],
                [5, 8],
                [5, 8],
                [5, 8],
                [5, 8],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 8],
                [6, 8],
                [6, 8],
                [6, 8],
                [6, 8],
                [6, 8],
                [6, 8],
                [7, 8],
                [7, 8],
                [7, 8],
                [7, 8],
                [7, 8],
                [7, 8],
                [7, 8],
            ]
        )
        assert np.allclose(nbrlist, expected_nbrlist)
        print(offsets)


if __name__ == "__main__":
    ut.main()
