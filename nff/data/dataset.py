import torch
import numbers
import numpy as np
from copy import deepcopy
from collections.abc import Iterable
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from nff.data import get_neighbor_list, get_bond_list
from nff.data.sparse import sparsify_tensor
import nff.utils.constants as const
import copy
import itertools


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

        props = concatenate_dict(self.props, other.props, stack=False)

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

    def generate_bonded_neighbor_list(self):

        """
        Generates a list of bonded neighbors in self.props. Uses nff.data.graphs.get_bonded_neighbors,
        which uses a set of bond distance cutoffs for different types of atom pairs.
        Args:
            None
        Returns:
            None
        """

        self.props['bonded_nbr_list'] = []
        old_pct = 0
        for i, nxyz in enumerate(self.props['nxyz']):

            new_pct = int(i/len(self.props["nxyz"])*100)
            if new_pct != old_pct:
                old_pct = new_pct
                print("{}% complete generating bonded neighbor list".format(new_pct))

            self.props['bonded_nbr_list'].append(torch.tensor(get_bond_list(nxyz)))

        # self.props['bonded_nbr_list'] = [torch.tensor(get_bond_list(nxyz))
        #     for nxyz in self.props['nxyz']]
        
    def set_degree_vec(self):

        """
        Sets the degree vector. This is a list of the number of atoms bonded to each atom in
        the dataset. Each different molecule gets its own degree vector.
        Args:
            None
        Returns:
            None 
        """

        self.props["degree_vec"] = []
        for num_atoms, bonded_nbr_list in zip(self.props["num_atoms"], self.props["bonded_nbr_list"]):

            # define A as an N x N zero matrix (N is the number of atoms)
            A = torch.zeros(num_atoms, num_atoms).to(torch.long)
            # every pair of indices with a bond then gets a 1
            A[bonded_nbr_list[:, 0], bonded_nbr_list[:, 1]] = 1
            # sum over one of the dimensions to get an array with bond number
            # for each atom
            d = A.sum(1)
            self.props["degree_vec"].append(d)

    def unique_pairs(self, bonded_nbr_list):

        """
        Reduces the bonded neighbor list to only include unique pairs of bonds. For example,
        if atoms 3 and 5 are bonded, then `bonded_nbr_list` will have items [3, 5] and also
        [5, 3]. This function will reduce the pairs only to [3, 5] (i.e. only the pair in which
        the first index is lower).

        Args:
            bonded_nbr_list (list): list of arrays of bonded pairs for each molecule.
        Returns:
            sorted_pairs (list): same as bonded_nbr_list but without duplicate pairs.

        """

        unique_pairs = []
        for pair in bonded_nbr_list:
            # sort according to the first item in the pair
            sorted_pair = torch.sort(pair)[0].numpy().tolist()
            if sorted_pair not in unique_pairs:
                unique_pairs.append(sorted_pair)

        # now make sure that the sorting is still good (this may be unnecessary but I added
        # it just to make sure)
        idx = list(range(len(unique_pairs)))
        # first_arg = list of the the first node in each pair
        first_arg = [pair[0] for pair in unique_pairs]
        # sorted_idx = sort the indices of unique_pairs by the first node in each pair
        sorted_idx = [item[-1] for item in sorted(zip(first_arg, idx))]
        # re-arrange by sorted_idx
        sorted_pairs = torch.LongTensor(np.array(unique_pairs)[sorted_idx])

        return sorted_pairs

    def set_bonds(self):

        """
        Set the bonds between atoms.
        Args:
            None
        Returns:
            None
        """

        self.props["bonds"] = []
        self.props["neighbors"] = []
        self.props["num_bonds"] = []

        old_pct = 0
        i = 0

        for bonded_nbr_list, degree_vec in zip(self.props["bonded_nbr_list"],
            self.props["degree_vec"]):

            i +=1

            new_pct = int(i/len(self.props["bonded_nbr_list"])*100)
            if new_pct != old_pct:
                old_pct = new_pct
                print("{}% complete bonds".format(new_pct))

            # get the unique set of bonded pairs
            bonds = self.unique_pairs(bonded_nbr_list)
            # neighbors is a list of bonded neighbor pairs for each atom.
            # Get it by splitting the bonded neighbor list by degree_vec
            neighbors = list(torch.split(bonded_nbr_list, degree_vec.tolist()))
            # second_arg_neighbors is just the second node in each set of bonded
            # neighbor pairs. Since the first node is already given implicitly by
            # the first index of `neighbors`, we don't need to use it anymore
            second_arg_neighbors = [neigbor[:, 1].tolist()
                                    for neigbor in neighbors]

            # props["bonds"] is the full set of unique pairs of bonded atoms
            self.props["bonds"].append(bonds)
            # props["num_bonds"] is teh number of bonds
            self.props["num_bonds"].append(torch.tensor(len(bonds)))
            # props["neighbors"] is the `second_arg_neighbors` intrdoduced above.
            # Note that props["neighbors"] is for bonded neighbors, as opposed
            # to props["nbr_list"], which is just everything within a 5 A radius.
            self.props["neighbors"].append(second_arg_neighbors)

    def set_angles(self):

        """
        Set the angles among bonded atoms.
        Args:
            None
        Returns:
            None

        """

        self.props["angles"] = []
        self.props["num_angles"] = []
        old_pct = 0

        for i, neighbors in enumerate(self.props["neighbors"]):

            new_pct = int(i/len(self.props["neighbors"])*100)
            if new_pct != old_pct:
                old_pct = new_pct
                print("{}% complete angles".format(new_pct))

            angles = [list(itertools.combinations(x, 2)) for x in neighbors]
            angles = [[[pair[0]]+[i]+[pair[1]] for pair in pairs]
                      for i, pairs in enumerate(angles)]
            angles = list(itertools.chain(*angles))
            angles = torch.LongTensor(angles)
            self.props["angles"].append(angles)
            self.props["num_angles"].append(torch.tensor(len(angles)))

    def set_dihedrals(self):

        """
        Set the dihedral angles among bonded atoms.
        Args:
            None
        Returns:
            None

        """

        self.props["dihedrals"] = []
        self.props["num_dihedrals"] = []
        old_pct = 0

        for i, neighbors in enumerate(self.props["neighbors"]):

            new_pct = int(i/len(self.props["neighbors"])*100)
            if new_pct != old_pct:
                old_pct = new_pct
                print("{}% complete dihedrals".format(new_pct))

            dihedrals = copy.deepcopy(neighbors)
            for i in range(len(neighbors)):
                for counter, j in enumerate(neighbors[i]):
                    k = set(neighbors[i])-set([j])
                    l = set(neighbors[j])-set([i])
                    pairs = list(
                        filter(lambda pair: pair[0] < pair[1], itertools.product(k, l)))
                    dihedrals[i][counter] = [[pair[0]]+[i]+[j]+[pair[1]]
                                             for pair in pairs]
            dihedrals = list(itertools.chain(
                *list(itertools.chain(*dihedrals))))
            dihedrals = torch.LongTensor(dihedrals)
            self.props["dihedrals"].append(dihedrals)
            self.props["num_dihedrals"].append(torch.tensor(len(dihedrals)))

    def set_impropers(self):

        """
        Set the improper angles among bonded atoms.
        Args:
            None
        Returns:
            None

        """
        self.props["impropers"] = []
        self.props["num_impropers"] = []

        for neighbors in self.props["neighbors"]:
            impropers = copy.deepcopy(neighbors)
            for i in range(len(impropers)):
                impropers[i] = [
                    [i]+list(x) for x in itertools.combinations(neighbors[i], 3)]
            impropers = list(itertools.chain(*impropers))
            impropers = torch.LongTensor(impropers)

            self.props["impropers"].append(impropers)
            self.props["num_impropers"].append(torch.tensor(len(impropers)))

    def set_pairs(self, use_1_4_pairs):

        """
        Set the non-bonded pairs.
        Args:
            None
        Returns:
            None

        """

        self.props["pairs"] = []
        self.props["num_pairs"] = []


        for i, neighbors in enumerate(self.props["neighbors"]):
            bonds = self.props["bonds"][i]
            angles = self.props["angles"][i]
            dihedrals = self.props["dihedrals"][i]
            impropers = self.props["impropers"][i]
            num_atoms = self.props["num_atoms"][i]

            pairs = torch.eye(num_atoms, num_atoms)
            topologies = [bonds, angles, impropers]

            if use_1_4_pairs is False:
                topologies.append(dihedrals)
            for topology in topologies:
                for interaction_list in topology:
                    for pair in itertools.combinations(interaction_list, 2):
                        pairs[pair[0], pair[1]] = 1
                        pairs[pair[1], pair[0]] = 1
            pairs = (pairs == 0).nonzero()
            pairs = pairs.sort(dim=1)[0].unique(dim=0).tolist()
            pairs = torch.LongTensor(pairs)

            self.props["pairs"].append(pairs)
            self.props["num_pairs"].append(torch.tensor(len(pairs)))

    def generate_topologies(self, use_1_4_pairs=True):

        """
        Generate bond, angle, dihedral, and improper topologies. Non-bonded pairs
        haven't been tested yet.
        Args:
            use_1_4_pairs (bool): whether or not to use 1-4 pairs when settings non-
                bonded pairs
        Returns:
            None
        """

        self.generate_bonded_neighbor_list()
        self.set_degree_vec()
        self.set_bonds()
        self.set_angles()
        self.set_dihedrals()
        self.set_impropers()
        # self.set_pairs(use_1_4_pairs)

        # remove props["neighbors"] because we don't need it anymore
        self.props.pop("neighbors")


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


def to_tensor(x, stack=False):
    """
    Converts input `x` to torch.Tensor

    Args:
        x: input to be converted. Can be: number, string, list, array, tensor
        stack (bool): if True, concatenates torch.Tensors in the batching dimension

    Returns:
        torch.Tensor or list, depending on the type of x
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

        if all([len(y.shape) <= 1 for y in x]):
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


def concatenate_dict(*dicts, stack=False):
    """Concatenates dictionaries as long as they have the same keys.
        If one dictionary has one key that the others do not have,
        the dictionaries lacking the key will have that key replaced by None.

    Args:
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
        stack (bool): if True, stacks the values when converting them to
            tensors.
    """

    assert all([type(d) == dict for d in dicts]), \
        'all arguments have to be dictionaries'

    keys = set(sum([list(d.keys()) for d in dicts], []))

    joint_dict = {}
    for key in keys:
        # flatten list of values
        values = []
        for d in dicts:
            num_values = len([x for x in d.values() if x is not None][0])
            val = d.get(key, to_tensor([np.nan] * num_values))
            if type(val) == list:
                values += val
            else:
                values.append(val)

        values = to_tensor(values, stack=stack)

        joint_dict[key] = values

    return joint_dict


def split_train_test(dataset, test_size=0.2):
    """Splits the current dataset in two, one for training and
        another for testing.
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

    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size / (1 - val_size))

    return train, validation, test

