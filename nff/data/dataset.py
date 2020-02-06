"""Summary
"""
import torch
import numbers
import numpy as np
import copy
import itertools
import nff.utils.constants as const
from copy import deepcopy
from collections.abc import Iterable
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from nff.data.sparse import sparsify_tensor
from nff.data.topology import update_props_topologies
from nff.data.graphs import reconstruct_atoms
from nff.io import AtomsBatch


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
        """Summary

        Returns:
            TYPE: Description
        """
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        """Summary

        Args:
            idx (TYPE): Description

        Returns:
            TYPE: Description
        """
        return {key: val[idx] for key, val in self.props.items()}

    def __add__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if other.units != self.units:
            other = other.copy().to_units(self.units)

        props = concatenate_dict(self.props, other.props)

        return Dataset(props, units=self.units)

    def _check_dictionary(self, props):
        """Check the dictionary or properties to see if it has the
        specified format.

        Args:
            props (TYPE): Description

        Returns:
            TYPE: Description
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
                props[key] = to_tensor([np.nan] * n_geoms)

            elif any([x is None for x in val]):
                bad_indices = [i for i, item in enumerate(val) if item is None]
                good_indices = [index for index in range(
                    len(val)) if index not in bad_indices]
                if len(good_indices) == 0:
                    nan_list = np.array([float("NaN")]).tolist()
                else:
                    good_index = good_indices[0]
                    nan_list = (np.array(val[good_index])
                                * float('NaN')).tolist()
                for index in bad_indices:
                    props[key][index] = nan_list
                props.update({key: to_tensor(val)})

            else:
                assert len(val) == n_geoms, \
                    'length of {} is not compatible with {} geometries'.format(
                        key, n_geoms)
                props[key] = to_tensor(val)

        return props

    def generate_neighbor_list(self, cutoff, undirected=True):
        """Generates a neighbor list for each one of the atoms in the dataset.
            By default, does not consider periodic boundary conditions.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
            undirected (bool, optional): Description

        Returns:
            TYPE: Description
        """
        self.props['nbr_list'] = [
            get_neighbor_list(nxyz[:, 1:4], cutoff, undirected)
            for nxyz in self.props['nxyz']
        ]

        return

    def copy(self):
        """Copies the current dataset

        Returns:
            TYPE: Description
        """
        return Dataset(self.props, self.units)

    def to_units(self, target_unit):
        """Converts the dataset to the desired unit. Modifies the dictionary of properties
            in place.

        Args:
            target_unit (str): unit to use as final one

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
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
        """Summary

        Returns:
            TYPE: Description
        """
        idx = list(range(len(self)))
        reindex = skshuffle(idx)
        self.props = {key: val[reindex] for key, val in self.props.items()}

        return

    def unwrap_xyz(self, mol_dic):
        """
        Unwrap molecular coordinates by displacing atoms by box vectors


        Args:
            mol_dic (dict): dictionary of nodes of each disconnected subgraphs
        """
        for i in range(len(self.props['nxyz'])):
            # makes atoms object
            atoms = AtomsBatch(positions=self.props['nxyz'][i][:, 1:4],
                               numbers=self.props['nxyz'][i][:, 0],
                               cell=self.props["cell"][i],
                               pbc=True
                               )

            # recontruct coordinates based on subgraphs index
            if self.props['smiles']:
                mol_idx = mol_dic[self.props['smiles'][i]]
                atoms.set_positions(reconstruct_atoms(atoms, mol_idx))
                nxyz = atoms.get_nxyz()
            self.props['nxyz'][i] = torch.Tensor(nxyz)

    def generate_topologies(self, bond_dic, use_1_4_pairs=True):
        """
        Generate topology for each Geom in the dataset.

        Args:
            bond_dic (dict): dictionary of bond lists for each smiles
            use_1_4_pairs (bool): consider 1-4 pairs when generating non-bonded neighbor list
        """
        # use the bond list to generate topologies for the props
        new_props = update_props_topologies(
            props=self.props, bond_dic=bond_dic, use_1_4_pairs=use_1_4_pairs)
        self.props = new_props

    def save(self, path):
        """Summary

        Args:
            path (TYPE): Description
        """
        torch.save(self, path)

    @classmethod
    def from_file(cls, path):
        """Summary

        Args:
            path (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            TypeError: Description
        """
        obj = torch.load(path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(
                '{} is not an instance from {}'.format(path, type(cls))
            )


def force_to_energy_grad(dataset):
    """
    Converts forces to energy gradients in a dataset. This conforms to
        the notation that a key with `_grad` is the gradient of the
        property preceding it. Modifies the database in-place.

    Args:
        dataset (TYPE): Description
        dataset (nff.data.Dataset)

    Returns:
        success (bool): if True, forces were removed and energy_grad
            became the new key.
    """
    if 'forces' not in dataset.props.keys():
        return False
    else:
        dataset.props['energy_grad'] = [
            -x
            for x in dataset.props.pop('forces')
        ]
        return True


def to_tensor(x, stack=False):
    """
    Converts input `x` to torch.Tensor.

    Args:
        x (list of lists): input to be converted. Can be: number, string, list, array, tensor
        stack (bool): if True, concatenates torch.Tensors in the batching dimension

    Returns:
        torch.Tensor or list, depending on the type of x

    Raises:
        TypeError: Description
    """

    # a single number should be a list
    if isinstance(x, numbers.Number):
        return torch.Tensor([x])

    if isinstance(x, str):
        return [x]

    if isinstance(x, torch.Tensor):
        return x

    # all objects in x are tensors
    if isinstance(x, list) and all([isinstance(y, torch.Tensor) for y in x]):

        # list of tensors with zero or one effective dimension
        # flatten the tensor

        if all([len(y.shape) < 1 for y in x]):
            return torch.cat([y.view(-1) for y in x], dim=0)

        elif stack:
            return torch.cat(x, dim=0)

        # list of multidimensional tensors
        else:
            return x

    # some objects are not tensors
    elif isinstance(x, list):

        # list of strings
        if all([isinstance(y, str) for y in x]):
            return x

        # list of ints
        if all([isinstance(y, int) for y in x]):
            return torch.LongTensor(x)

        # list of floats
        if all([isinstance(y, numbers.Number) for y in x]):
            return torch.Tensor(x)

        # list of arrays or other formats
        if any([isinstance(y, (list, np.ndarray)) for y in x]):
            return [torch.Tensor(y) for y in x]

    raise TypeError('Data type not understood')


def concatenate_dict(*dicts):
    """Concatenates dictionaries as long as they have the same keys.
        If one dictionary has one key that the others do not have,
        the dictionaries lacking the key will have that key replaced by None.

    Args:
        *dicts: Description
        *dicts (any number of dictionaries)
            Example:
                dict_1 = {
                    'nxyz': [...],
                    'energy': [...]
                }
                dict_2 = {
                    'nxyz': [...],
                    'energy': [...]
                }
                dicts = [dict_1, dict_2]

    Returns:
        TYPE: Description

    """

    assert all([type(d) == dict for d in dicts]), \
        'all arguments have to be dictionaries'

    keys = set(sum([list(d.keys()) for d in dicts], []))

    def get_length(value):
        if isinstance(value, list):
            return len(value)
        return 1

    def get_length_of_values(dict_):
        return min([get_length(v) for v in dict_.values()])

    def flatten_val(value):
        """Given a value, which can be a number, a list or
            a torch.Tensor, return its flattened version
            to be appended to a list of values
        """
        if isinstance(value, list):
            return value

        elif get_length(value) == 1:
            return [value]
        
        return value

    # we have to see how many values the properties of each dictionary has.
    values_per_dict = [get_length_of_values(d) for d in dicts]

    # creating the joint dicionary
    joint_dict = {}
    for key in keys:
        # flatten list of values
        values = []
        old_values = values_per_dict[0]
        for num_values, d in zip(values_per_dict, dicts):
            # if the dictionary does not have that key, we replace that with None
            val = d.get(
                key,
                [None] * num_values if num_values > 1 else None
            )

            if (hasattr(val, '__len__') and len(val) == 1) or (num_values == 1 and old_values != 1):
                values.append(val)
            else:
                values.append([val] if num_values == 1 else val)

            old_values = num_values


        new_values = []
        subitem_0 = values[0][0]
        for i, sublist in enumerate(values):
            for subitem in sublist:
                if type(subitem) is torch.Tensor: 
                    new_values.append(subitem.reshape(-1, *subitem_0.shape[1:] ))
                else:
                    new_values.append(subitem)

        joint_dict[key] = new_values

    return joint_dict


def split_train_test(dataset, test_size=0.2):
    """Splits the current dataset in two, one for training and
    another for testing.

    Args:
        dataset (TYPE): Description
        test_size (float, optional): Description

    Returns:
        TYPE: Description
    """

    idx = list(range(len(dataset)))
    idx_train, idx_test = train_test_split(idx, test_size=test_size)
    
    props = {key: [val[i] for i in idx_train]
             for key, val in dataset.props.items()}

    train = Dataset(
        props={key: [val[i] for i in idx_train]
               for key, val in dataset.props.items()},
        units=dataset.units
    )
    test = Dataset(
        props={key: [val[i] for i in idx_test]
               for key, val in dataset.props.items()},
        units=dataset.units
    )

    return train, test


def split_train_validation_test(dataset, val_size=0.2, test_size=0.2):
    """Summary

    Args:
        dataset (TYPE): Description
        val_size (float, optional): Description
        test_size (float, optional): Description

    Returns:
        TYPE: Description
    """
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size / (1 - val_size))

    return train, validation, test
