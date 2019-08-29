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
                 props,
                 units='atomic'):
        """Constructor for Dataset class.

        Args:
            props (dictionary of lists): dictionary containing the
                properties of the system. Each key has a list, and 
                all lists have the same length.
            units (str): units of the system.
        """

        self._check_dictionary(props)
        n_atoms = len(props['nxyz'])

        if 'pbc' not in props:
            props['pbc'] = [None] * len(nxyz)

        self.nxyz = self._to_array(nxyz)
        self.units = units
        self.to_units('kcal/mol')

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}

    def __add__(self, other):
        if other.units != self.units:
            print('changing units')
            other.to_units(self.units)
        
        props = concatenate_dict(self.props, other.props)

        return Dataset(props, units=self.units)

    def _check_dictionary(self, props):
        """Check the dictionary or properties to see if it has the
            specified format.
        """

        assert 'nxyz' in props.keys()
        n_atoms = len(props['nxyz'])

        for key, val in props.items():
            if val is None:
                props[key] = [None] * n_atoms
            
            assert len(val) == n_atoms, 'length of {} is not {}'.format(key, n_atoms)

    def _to_array(self, x):
        """Converts input `x` to array"""
        array = np.array
        
        if type(x[0]) == float:
            return array(x)
        else:
            return [array(_) for _ in x]

    def to_units(self, target_unit):
        if target_unit == 'kcal/mol' and self.units == 'atomic':
            self.to_kcal_mol()

        elif target_unit == 'atomic' and self.units == 'kcal/mol':
            self.to_atomic_units()

        else:
            pass

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
        idx = list(range(len(self)))
        reindex = skshuffle(idx)
        self.props = self[reindex]
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


def concatenate_dict(*dicts):
    """Concatenates dictionaries as long as they have the same keys.
        If one dictionary has one key that the others do not have,
        the dictionaries lacking the key will have that key replaced by None. All dictionaries must have the key `nxyz`.

    Args:
        *dicts (any number of dictionaries)
    """

    assert all([type(d) == dict for d in dicts]), \
        'all arguments have to be dictionaries'

    keys = set(sum([list(d.keys()) for d in dicts], []))
    
    joint_dict = {}
    for key in keys:
        joint_dict[key] = [
            d.get(key, [None] * len(d['nxyz']))
            for d in dicts
        ]

    return joint_dict


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
        nxyz=nxyz_train,
        energy=energy_train,
        force=force_train,
        smiles=smiles_train,
        pbc=pbc_train,
        units=dataset.units
    )

    test = Dataset(
        nxyz=nxyz_test,
        energy=energy_test,
        force=force_test,
        smiles=smiles_test,
        pbc=pbc_test,
        units=dataset.units
    )

    return train, test


def split_train_validation_test(dataset, val_size=0.2, test_size=0.2):
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size)

    return train, validation, test
