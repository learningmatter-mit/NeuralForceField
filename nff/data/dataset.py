import torch 
import numpy as np 

from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from nff.data import GraphDataset
import nff.utils.constants as const


class Dataset(TorchDataset):
    """Dataset to deal with NFF calculations. Can be expanded to retrieve calculations
         from the cluster later.

    Attributes:
        nxyz (array): (N, 4) array with atomic number and xyz coordinates
            for each of the N atoms
        energy (array): (N, ) array with energies
        force (array): (N, 3) array with forces
        smiles (array): (N, ) array with SMILES strings
    """

    def __init__(self,
                 nxyz,
                 energy,
                 force,
                 smiles,
                 pbc=None,
                 atomic_units=False):
        """Constructor for Dataset class.

        Args:
            nxyz (array): (N, 4) array with atomic number and xyz coordinates
                for each of the N atoms
            energy (array): (N, ) array with energies
            force (array): (N, 3) array with forces
            smiles (array): (N, ) array with SMILES strings
            pbc (array): array with indices for periodic boundary conditions
            atomic_units (bool): if True, input values are given in atomic units.
                They will be converted to kcal/mol.
        """

        if pbc is None:
            pbc = [None] * len(nxyz)

        assert all([
            len(_) == len(nxyz)
            for _ in [energy, force, smiles, pbc]
        ]), 'All lists should have the same length.'

        self.nxyz = self._to_array(nxyz)
        self.energy = energy
        self.force = self._to_array(force)
        self.smiles = smiles
        self.pbc = pbc

        self.units = 'atomic' if atomic_units else 'kcal/mol'

        if atomic_units:
            self.to_kcal_mol()

    def __len__(self):
        return len(self.nxyz)

    def __getitem__(self, idx):
        return self.nxyz[idx], self.energy[idx], self.force[idx], self.smiles[idx], self.pbc[idx]

    def _to_array(self, x):
        """Converts input `x` to array"""
        array = np.array
        
        if type(x[0]) == float:
            return array(x)
        else:
            return [array(_) for _ in x]

    def to_kcal_mol(self):
        """Converts forces and energies from atomic units to kcal/mol."""

        self.force = [
            x * const.HARTREE_TO_KCAL_MOL / const.BOHR_RADIUS
            for x in self.force
        ]
        self.energy = [x * const.HARTREE_TO_KCAL_MOL for x in self.energy]

        self.units = 'kcal/mol'

    def to_atomic_units(self):
        self.force = [
            x / const.HARTREE_TO_KCAL_MOL * const.BOHR_RADIUS
            for x in self.force
        ]
        self.energy = [x / const.HARTREE_TO_KCAL_MOL for x in self.energy]
        self.units = 'atomic'

    def shuffle(self):
        self.nxyz, self.force, self.energy, self.smiles, self.pbc = skshuffle(
            self.nxyz, self.force, self.energy, self.smiles, self.pbc
        )
        return 

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def from_file(cls, path):
        obj = torch.load(path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(
                '{} is not an instance from {}'.format(path, type(cls))
            )


def split_train_test(dataset, test_size=0.2):
    """Splits the current dataset in two, one for training and
        another for testing.
    """

    (
        nxyz_train, nxyz_test,
        energy_train, energy_test,
        force_train, force_test,
        smiles_train, smiles_test,
        pbc_train, pbc_test
    ) = train_test_split(
        dataset.nxyz,
        dataset.energy,
        dataset.force,
        dataset.smiles,
        dataset.pbc,
        test_size=test_size
    )

    train = Dataset(
        nxyz_train,
        energy_train,
        force_train,
        smiles_train,
        pbc_train
    )

    test = Dataset(
        nxyz_test,
        energy_test,
        force_test,
        smiles_test,
        pbc_test
    )

    return train, test


def split_train_validation_test(dataset, val_size=0.2, test_size=0.2):
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size)

    return train, validation, test
