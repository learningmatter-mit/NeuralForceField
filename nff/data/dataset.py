import torch 
import numpy as np 

from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from nff.data import Graph, GraphDataset
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

    array_type = np.array

    def __init__(self,
                 nxyz,
                 energy,
                 force,
                 smiles,
                 atomic_units=False):
        """Constructor for Dataset class.

        Args:
            nxyz (array): (N, 4) array with atomic number and xyz coordinates
                for each of the N atoms
            energy (array): (N, ) array with energies
            force (array): (N, 3) array with forces
            smiles (array): (N, ) array with SMILES strings
            atomic_units (bool): if True, input values are given in atomic units.
                They will be converted to kcal/mol.
        """

        assert all([
            len(_) == len(nxyz)
            for _ in [energy, force, smiles]
        ]), 'All lists should have the same length.'

        self.nxyz = self.array_type(nxyz)
        self.energy = self.array_type(energy)
        self.force = self.array_type(force)
        self.smiles = self.array_type(smiles)

        if atomic_units:
            self.units_to_kcal_mol()

    def __len__(self):
        return len(self.nxyz)

    def __getitem__(self, idx):
        return self.nxyz[idx], self.energy[idx], self.force[idx], self.smiles[idx]
    
    def to_kcal_mol(self):
        """Converts forces and energies from atomic units to kcal/mol."""
    
        self.force = self.force * const.HARTREE_TO_KCAL_MOL / const.BOHR_RADIUS
        self.energy = self.energy * const.HARTREE_TO_KCAL_MOL 

    def to_atomic_units(self):
        self.force = self.force / const.HARTREE_TO_KCAL_MOL * const.BOHR_RADIUS
        self.energy = self.energy / const.HARTREE_TO_KCAL_MOL
    
    def shuffle(self):
        self.nxyz, self.force, self.energy, self.smiles = skshuffle(
            self.nxyz, self.force, self.energy, self.smiles
        )
        return 


def split_train_test(dataset, test_size=0.2):
    """Splits the current dataset in two, one for training and
        another for testing.
    """

    (
        nxyz_train, nxyz_test,
        energy_train, energy_test,
        force_train, force_test,
        smiles_train, smiles_test
    ) = train_test_split(
        dataset.nxyz,
        dataset.energy,
        dataset.force,
        dataset.smiles,
        test_size=test_size
    )

    train = Dataset(
        nxyz_train,
        energy_train,
        force_train,
        smiles_train
    )

    test = Dataset(
        nxyz_test,
        energy_test,
        force_test,
        smiles_test
    )

    return train, test


def split_train_test_validation(dataset, test_size=0.2, val_size=0.2):
    train, test = split_train_test(dataset, test_size=test_size)
    train, validation = split_train_test(train, val_size=val_size)

    return train, test, validation
