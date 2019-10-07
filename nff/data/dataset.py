import torch 
import numpy as np 
from copy import deepcopy
from collections.abc import Iterable

from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from nff.data import get_neighbor_list
from nff.data.sparse import sparsify_tensor
import nff.utils.constants as const
import copy


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
                    'energy_0_grad': [np.array([[0, 0, 0], [0.1, 0.2, 0.3]]), np.array([[0, 0, 0], [0.1, 0.2, 0.3]])],
                    'energy_1': [1.5, 1.5],
                    'energy_1_grad': [np.array([[0, 0, 1], [0.1, 0.5, 0.8]]), np.array([[0, 0, 1], [0.1, 0.5, 0.8]])],
                    'dipole_2': [3, None]
                }

            Periodic boundary conditions must be specified through the 'offset' key in props.
                Once the neighborlist is created, distances between
                atoms are computed by subtracting their xyz coordinates
                and adding to the offset vector. This ensures images
                of atoms outside of the unit cell have different
                distances when compared to atoms inside of the unit cell.
                This also bypasses the need for a reindexing.

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
            props['num_atoms'] = torch.LongTensor(n_atoms)
        else:
            props['num_atoms'] = torch.LongTensor(props['num_atoms'])

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
            By default, does not consider periodic boundary conditions.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
        """
        self.props['nbr_list'] = [
            get_neighbor_list(nxyz[:, 1:4], cutoff)
            for nxyz in self.props['nxyz']
        ]

        # self.props['offsets'] = [0] * len(self)

        return

    def copy(self):
        """Copies the current dataset"""
        return Dataset(self.props, self.units)

    def to_units(self, target_unit):
        """Converts the dataset to the desired unit. Modifies the dictionary of properties
            in place.

        Args:
            target_unit (str): unit to use as final one
        """

        if target_unit not in ['kcal/mol', 'atomic']:
            raise NotImplementedError(
                'unit conversion for {} not implemented'.format(target_unit)
            )

        if target_unit == 'kcal/mol' and self.units == 'atomic':
            self.props = const.convert_units(
                self.props,
                const.AU_TO_KCAL
            )

        elif target_unit == 'atomic' and self.units == 'kcal/mol':
            self.props = const.convert_units(
                self.props,
                const.KCAL_TO_AU
            )
        else:
            return

        self.units = target_unit
        return

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
        props={key: [val[i] for i in idx_train] for key, val in dataset.props.items()},
        units=dataset.units
    )
    test = Dataset(
        props={key: [val[i] for i in idx_test] for key, val in dataset.props.items()},
        units=dataset.units
    )

    return train, test


def split_train_validation_test(dataset, val_size=0.2, test_size=0.2):
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size / (1 - val_size))

    return train, validation, test
