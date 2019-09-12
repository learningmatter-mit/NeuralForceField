import torch 
import numpy as np 
from copy import deepcopy
from collections.abc import Iterable

from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from nff.data import get_neighbor_list
import nff.utils.constants as const

class Dataset(TorchDataset):
    """Dataset to deal with NFF calculations. Can be expanded to retrieve calculations
         from the cluster later.

    Attributes:
        props (list of dicts): list of dictionaries containing all properties of the system.
            Keys are the name of the property and values are the properties. Each value
            is given by `props[idx][key]`. The only mandatory key is 'nxyz'. If inputting
            energies, forces or hessians of different electronic states, the quantities 
            should be distinguished with a "_n" suffix, where n = 0, 1, 2, ...
            Whatever name is given to the energy of state n, the corresponding force name
            must be the exact same name, but with "energy" replaced by "force".

            Example:

                props = {
                    'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]), np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                    'energy_0': [1, 1.2],
                    'force_0': [np.array([[0, 0, 0], [0.1, 0.2, 0.3]]), np.array([[0, 0, 0], [0.1, 0.2, 0.3]])],
                    'energy_1': [1.5, 1.5],
                    'force_0': [np.array([[0, 0, 1], [0.1, 0.5, 0.8]]), np.array([[0, 0, 1], [0.1, 0.5, 0.8]])],
                    'dipole_2': [3, None]
                }

        units (str): units of the energies, forces etc.

    """

    def __init__(self,
                 props,
                 units='kcal/mol'):
        """Constructor for Dataset class.

        Args:
            props (dictionary of lists): dictionary containing the
                properties of the system. Each key has a list, and 
                all lists have the same length.
            units (str): units of the system.
        """

        self.props = self._check_dictionary(deepcopy(props))
        self.units = units
        self.to_units('kcal/mol')

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}

    def __add__(self, other):
        if other.units != self.units:
            other = other.copy().to_units(self.units)
        
        props = concatenate_dict(self.props, other.props)

        return Dataset(props, units=self.units)

    def _check_dictionary(self, props):
        """Check the dictionary or properties to see if it has the
            specified format.
        """

        assert 'nxyz' in props.keys()
        n_atoms = [len(x) for x in props['nxyz']]
        n_geoms = len(props['nxyz'])

        if 'num_atoms' not in props.keys():
            props['num_atoms'] = n_atoms
        else:
            assert props['num_atoms'] == n_atoms

        for key, val in props.items():

            if val is None:
                props[key] = self._to_array([np.nan] * n_geoms)

            elif any([x is None for x in val]):
                bad_indices = [i for i, item in enumerate(val) if item is None]
                good_index = [index for index in range(len(val)) if index not in bad_indices][0]
                nan_list = (np.array(val[good_index]) * float('NaN')).tolist()
                for index in bad_indices:
                    props[key][index] = nan_list
                props.update({key: self._to_array(val)})

            else:
                assert len(val) == n_geoms, \
                    'length of {} is not compatible with {} geometries'.format(key, n_geoms)
                props[key] = self._to_array(val)

        if 'pbc' not in props:
            props['pbc'] = [torch.LongTensor(range(x)) for x in n_atoms]

        return props

    def _to_array(self, x):
        """Converts input `x` to array"""
        array = torch.Tensor
        if isinstance(x[0], (array, str)):
            return x
        
        if isinstance(x[0], Iterable):
            return [array(_) for _ in x]
        else:
            return array(x)

    def generate_neighbor_list(self, cutoff):
        """Generates a neighbor list for each one of the atoms in the dataset.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
        """
        self.props['nbr_list'] = [
            get_neighbor_list(nxyz[:, 1:4], cutoff)
            for nxyz in self.props['nxyz']
        ]
        return

    def copy(self):
        """Copies the current dataset"""
        return Dataset(self.props, self.units)

    def to_units(self, target_unit):
        if target_unit == 'kcal/mol' and self.units == 'atomic':
            self.to_kcal_mol()

        elif target_unit == 'atomic' and self.units == 'kcal/mol':
            self.to_atomic_units()

        else:
            pass

    def to_kcal_mol(self): 
        """Converts forces and energies from atomic units to kcal/mol."""

        for key in self.props.keys():
            if "energy" in key:  
                self.props[key] = [x * const.HARTREE_TO_KCAL_MOL for x in self.props[key]]
            elif "force" in key:
                self.props[key] = [
                    x * const.HARTREE_TO_KCAL_MOL / const.BOHR_RADIUS
                    for x in self.props[key]
                ]

        self.units = 'kcal/mol'

    def to_atomic_units(self):
        for key in self.props.keys():
            if "energy" in key:
                self.props[key] = [x / const.HARTREE_TO_KCAL_MOL for x in self.props[key]]
            if "force" in key:
                self.props[key] = [
                    x / const.HARTREE_TO_KCAL_MOL * const.BOHR_RADIUS
                    for x in self.props['force']
                ]

        self.units = 'atomic'

    def shuffle(self):
        idx = list(range(len(self)))
        reindex = skshuffle(idx)
        self.props = {key: val[reindex] for key, val in self.props.items()}

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

    idx = list(range(len(dataset)))
    idx_train, idx_test = train_test_split(idx)
    train = Dataset(
        props=slice_props_by_idx(idx_train, dataset.props),
        units=dataset.units
    )
    test = Dataset(
        props=slice_props_by_idx(idx_test, dataset.props),
        # This is buggy for me (python 3.5.6): {key: val[idx_test] for key, val in dataset.props.items()}
        units=dataset.units
    )

    return train, test

def slice_props_by_idx(idx, dictionary):
    """for a dicionary of lists, build a new dictionary given index
    
    Args:
        idx (list): Description
        dictionary (dict): Description
    
    Returns:
        dict: sliced dictionary
    """
    props_dict = {}
    for key, val in dictionary.items(): 
        val_list = []
        for i in idx:
            val_list.append(val[i])
        props_dict[key] = val_list
    return props_dict


def split_train_validation_test(dataset, val_size=0.2, test_size=0.2):
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size)

    return train, validation, test
