import torch 
import numbers
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
import itertools


import pdb


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
        # pdb.set_trace()
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


        # new_props = {}

        # for key, val in props.items():

        #     new_val = []

        #     if all([type(x) == torch.Tensor for x in val]) and (
        #         any([len(x.shape) != 0 and x.shape[0] == 0 for x in val])):

        #         for i, x in enumerate(copy.deepcopy(val)):
        #             if x.shape[0] == 0:
        #                 new_val.append(None)
        #             else:
        #                 new_val.append(val[i])
        #     else:
        #         new_val = copy.deepcopy(val)
        #     new_props[key] = new_val

        # props = copy.deepcopy(new_props)

        for key, val in props.items():

            if val is None:
                props[key] = to_tensor([np.nan] * n_geoms)


            elif any([x is None for x in val]):
                bad_indices = [i for i, item in enumerate(val) if item is None]
                good_indices = [index for index in range(len(val)) if index not in bad_indices]
                if len(good_indices) == 0:
                    nan_list = np.array([float("NaN")]).tolist()
                else:
                    good_index = good_indices[0]
                    nan_list = (np.array(val[good_index]) * float('NaN')).tolist()
                for index in bad_indices:
                    props[key][index] = nan_list
                props.update({key: to_tensor(val)})

            else:
                assert len(val) == n_geoms, \
                    'length of {} is not compatible with {} geometries'.format(key, n_geoms)
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

    def set_degree_vec(self):

        self.props["degree_vec"] = []
        for num_atoms, nbr_list in zip(self.props["num_atoms"], self.props["nbr_list"]):

            A = torch.zeros(num_atoms, num_atoms).to(torch.long)
            # every pair of indices with a bond then gets a 1
            A[nbr_list[:, 0], nbr_list[:, 1]] = 1
            d = A.sum(1)
            self.props["degree_vec"].append(d)


    def unique_pairs(self, nbr_list):

        # pdb.set_trace()

        unique_pairs = []
        for pair in nbr_list:
            sorted_pair = torch.sort(pair)[0].numpy().tolist()
            if sorted_pair not in unique_pairs:
                unique_pairs.append(sorted_pair)

        idx = list(range(len(unique_pairs)))
        first_arg = [pair[0] for pair in unique_pairs]
        sorted_idx = [item[-1] for item in sorted(zip(first_arg, idx))]
        sorted_pairs = torch.LongTensor(np.array(unique_pairs)[sorted_idx])

        return sorted_pairs
        
    def set_bonds(self):

        # pdb.set_trace()

        self.props["bonds"] = []
        self.props["neighbors"] = []
        self.props["num_bonds"] = []

        for nbr_list, degree_vec in zip(self.props["nbr_list"], self.props["degree_vec"]):

            bonds = self.unique_pairs(nbr_list)
            neighbors = list(torch.split(nbr_list, degree_vec.tolist()))
            second_arg_neighbors = [neigbor[:,1].tolist() for neigbor in neighbors]

            self.props["bonds"].append(bonds)
            self.props["num_bonds"].append(torch.tensor(len(bonds)))
            self.props["neighbors"].append(second_arg_neighbors)



    def set_angles(self):

        # pdb.set_trace()

        self.props["angles"] = []
        self.props["num_angles"] = []

        for neighbors in self.props["neighbors"]:

            angles = [list(itertools.combinations(x, 2)) for x in neighbors]
            angles = [[[pair[0]]+[i]+[pair[1]] for pair in pairs] for i, pairs in enumerate(angles)]
            angles = list(itertools.chain(*angles))
            angles = torch.LongTensor(angles)
            self.props["angles"].append(angles)
            self.props["num_angles"].append(torch.tensor(len(angles)))


    def set_dihedrals(self):

        # pdb.set_trace()

        self.props["dihedrals"] = []
        self.props["num_dihedrals"] = []

        for neighbors in self.props["neighbors"]:
            dihedrals = copy.deepcopy(neighbors)
            for i in range(len(neighbors)):
                for counter, j in enumerate(neighbors[i]):
                    k = set(neighbors[i])-set([j])
                    l = set(neighbors[j])-set([i])
                    pairs = list(filter(lambda pair: pair[0]<pair[1], itertools.product(k, l)))
                    dihedrals[i][counter] = [[pair[0]]+[i]+[j]+[pair[1]] for pair in pairs]
            dihedrals = list(itertools.chain(*list(itertools.chain(*dihedrals))))
            dihedrals = torch.LongTensor(dihedrals)
            self.props["dihedrals"].append(dihedrals)
            self.props["num_dihedrals"].append(torch.tensor(len(dihedrals)))


    def set_impropers(self):

        # pdb.set_trace()


        self.props["impropers"] = []
        self.props["num_impropers"] = []

        for neighbors in self.props["neighbors"]:
            impropers = copy.deepcopy(neighbors)
            for i in range(len(impropers)):
                impropers[i] = [[i]+list(x) for x in itertools.combinations(neighbors[i], 3)]
            impropers = list(itertools.chain(*impropers))
            impropers = torch.LongTensor(impropers)

            self.props["impropers"].append(impropers)
            self.props["num_impropers"].append(torch.tensor(len(impropers)))


    def set_pairs(self, use_1_4_pairs):

        # pdb.set_trace()

        self.props["pairs"] = []
        self.props["num_pairs"] = []

        # pdb.set_trace()
        
        for i, neighbors in enumerate(self.props["neighbors"]):
            bonds = self.props["bonds"][i]
            angles = self.props["angles"][i]
            dihedrals = self.props["dihedrals"][i]
            impropers = self.props["impropers"][i]
            num_atoms = self.props["num_atoms"][i]

            pairs = torch.eye(num_atoms, num_atoms)
            topologies = [bonds, angles, impropers]

            # pdb.set_trace()
            # print(i)

            if use_1_4_pairs is False:
                topologies.append(dihedrals)
            for topology in topologies:
                for interaction_list in topology:
                    for pair in itertools.combinations(interaction_list, 2):
                        pairs[pair[0],pair[1]] = 1
                        pairs[pair[1],pair[0]] = 1
            pairs = (pairs == 0).nonzero()
            pairs = pairs.sort(dim=1)[0].unique(dim=0).tolist()
            pairs = torch.LongTensor(pairs)

            # pdb.set_trace()

            self.props["pairs"].append(pairs)
            self.props["num_pairs"].append(torch.tensor(len(pairs)))


    def generate_topologies(self, use_1_4_pairs=True):

        # pdb.set_trace()

        self.set_degree_vec()
        self.set_bonds()
        self.set_angles()
        self.set_dihedrals()
        self.set_impropers()

        self.props.pop("neighbors")

        # self.props["neighbors"] = [torch.LongTensor(neighbor) for
        #                            neighbor in self.props["neighbors"]]
        # self.set_pairs(use_1_4_pairs)

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

    # must deal with the case that some or all of the constituent tensors are empty, 
    # and/or they have different shapes


    # all objects in x are tensors
    if isinstance(x, list) and all([isinstance(y, torch.Tensor) for y in x]):

        # if some are empty, just return the list
        # pdb.set_trace()
        # shapes = [y.shape for y in x]
        # if any([len(shape) == 0 for shape in shapes]):
        #     return x

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

        # list of numbers
        if all([isinstance(y, numbers.Number) for y in x]):
            return torch.Tensor(x)

        # list of arrays or other formats
        pdb.set_trace()
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

    # pdb.set_trace()
    props={key: [val[i] for i in idx_train] for key, val in dataset.props.items()}

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

    # pdb.set_trace()
    train, validation = split_train_test(dataset, test_size=val_size)
    train, test = split_train_test(train, test_size=test_size / (1 - val_size))

    return train, validation, test






# #######################################################################################
# #######################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################

# # from geom object in pgmols

# def get_coords(self):
#     return [dict(element=PERIODICTABLE.GetElementSymbol(int(l[0])),
#                  x=l[1],
#                  y=l[2],
#                  z=l[3]
#                  ) for l in self.xyz]

# def set_coords(self, coord_list):
#     self.xyz = []
#     for atom_dict in coord_list:
#         row = [PERIODICTABLE.GetAtomicNumber(str(atom_dict['element'])),
#                atom_dict['x'],
#                atom_dict['y'],
#                atom_dict['z']]
#         self.xyz.append(row)

# def as_xyz(self):
#     coords = self.get_coords()
#     output = str(len(coords)) + "\n\n"
#     for c in coords:
#         output += " ".join([c["element"], str(c["x"]),
#                             str(c["y"]), str(c["z"])]) + "\n"
#     return output

# def distancemat(self):
#     xyz = np.array(self.xyz)[:, 1:]
#     atomicnums = np.array(self.xyz)[:, 0].astype(int)
#     return atomicnums, cdist(xyz, xyz)

# def adjdistmat(self, threshold=DISTANCETHRESHOLD):
#     atomicnums, dmat = self.distancemat()
#     if type(threshold) is dict:
#         thresholdmat = np.array([[threshold.get(tuple(
#             sorted((i, j))), DISTANCETHRESHOLD) for i in atomicnums] for j in atomicnums])
#         adjmat = (dmat < thresholdmat).astype(int)
#     else:
#         adjmat = (dmat < threshold).astype(int)
#     np.fill_diagonal(adjmat, 0)
#     return np.array(atomicnums), np.array(adjmat), np.array(dmat)


# #######################################################################################
# #######################################################################################

# def get_nuclei_bond_and_bondlen(geom):
#     """get list of bond, atomicnums, bondlength
    
#     Args:
#         geom (Geom): Description
    
#     Returns:
#         list: bondlist, atomic numbers, bond length
#     """
#     atomicnums, adjmat, dmat = geom.adjdistmat(threshold=DISTANCETHRESHOLDICT)
#     mol_bond = np.array(adjmat.nonzero()).transpose()#.tolist()
#     mol_nuclei = [atomicnums.tolist()]
#     mol_bondlen = dmat[mol_bond[:, 0], mol_bond[:, 1]]
    
#     return [mol_bond.tolist(), mol_nuclei, mol_bondlen.tolist()]


# def check_graph_reference_existence(smileslist, methodname='molecular_mechanics_mmff94', projectname='lipoly'):
#     """check if the clusters or molecules in the database has a converged reference structure
    
#     Args:
#         smileslist (list): list of smiles
#         methodname (str, optional): method name
#         projectname (str, optional): project name
    
#     Returns:
#         TYPE: smiles that have a reference graph, smiles that do not
#     """
#     graph_ref_not_exist_smiles = []
#     graph_ref_exist_smiles = []
#     for smiles in smileslist:
#         rootsp = Species.objects.filter(smiles=smiles,group__name=projectname).first().components.all()
        
#         if rootsp.count() == 0:
#             graph_ref_exist_smiles.append(smiles)
        
#         for sp in rootsp:
#             converged_geom_count = Geom.objects.filter(species=sp, 
#                                                          converged=True, 
#                                                          method__name=methodname
#                                                         ).count()
#             if converged_geom_count == 0:
#                 graph_ref_not_exist_smiles.append(smiles)
#             else:
#                 graph_ref_exist_smiles.append(smiles)

#             else:
#                 graph_ref_exist_smiles.append(smiles)
                
#     return  graph_ref_exist_smiles, graph_ref_not_exist_smiles

# def get_mol_ref(smileslist, 
#                 groupname='lipoly',
#                 method_name='molecular_mechanics_mmff94'):
#     '''
#         Obtain adjacency matrix and reference node order from goems 
        
#         To do: bond list should only be stored once
#     '''
    
#     # check first if all the smiles has a valid reference graph given the method
#     assert type(smileslist) == list
#     smileslist, nographlist = check_graph_reference_existence(smileslist, methodname=method_name, projectname=groupname)
#     if nographlist != []:
#         raise Exception("{} has no reference graph for method name {} ".format(''.join(nographlist), method_name))
    

#     mol_ref = dict()
#     species = Species.objects.filter(group__name=groupname, smiles__in=smileslist)
#     method = Method.objects.filter(name=method_name).first()
    
#     # bond list is stored twice as directed edge 
#     for sp in species:
#         if sp.smiles not in mol_ref:  
#             geom = Geom.objects.filter(species=sp
#                                       ).order_by("calcs__props__totalenergy").first()
#             # getting reference geoms as query sets 
#             # What method should we use as default parent geoms 
            
#             if '.' in sp.smiles: # This is a molecular cluster 
#                 rootparentgeoms = getrootconformers(geom=geom, method=method)
#                 # Use a function to get species count as a set                
#                 parentrepeatsdictionary = gen_geomsrepeats_in_cluster(geom, rootparentgeoms)

#                 # Generate reference nuclei list and its correponding molecular graph bond list 
#                 cluster_nuclei, cluster_bond, cluster_bondlen = gen_cluster_superlist(parentrepeatsdictionary)

#                 # update mol_ref dictionary 
#                 mol_ref[sp.smiles] = [cluster_bond, cluster_nuclei, cluster_bondlen]
#             else:              
#                 mol_ref[sp.smiles] = get_nuclei_bond_and_bondlen(geom)
        
#     return mol_ref


# def list2adj(bond_list, size=None):
#     E = bond_list
#     if size is None:
#         size = max(set([n for e in E for n in e])) + 1
#     adjacency = [[0]*size for _ in range(size)]
#     for sink, source in E:
#         adjacency[sink][source] = 1
#     return(adjacency)


# mol_ref = get_mol_ref(groupname=job_details.group, smileslist=get_smiles_list(geom_ids))
# mol_ref = mol_ref[geom.species.smiles]
# ref_bond_list, ref_node_order, ref_bond_len = mol_ref
# geom_data.A = list2adj(ref_bond_list, size=num_atoms)
# geom_data.xyz = np.array(geom.xyz)
















# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################

# class Graph():

#     """
#     Graph() is a flexible class for constructing labeled directed graphs (no self-loops and no parallel edges) using pytorch tensors
#     Includes (optional) functionality for handling graphs with three-dimensional structure
#     Takes pytorch tensors as input and produces pytorch tensors as output
#     Can be utilized within a pytorch computational graph (that is, gradients can be calculated with respect to certain input tensors)
#     Includes several utility functions for labeling nodes, edges, and the entire graph
#     To ensure accurate behavior, you should create and manipulate graphs using the provided functions rather than manipulating class fields directly
#     """

#     def __init__(self, N=None, A=None, pbc=None, dynamic=False):
#         """

#         A graph is initialized by providing either N (the number of nodes) or A (the adjacency matrix)
#         A should be provided as a 2D torch.LongTensor (the full adjacency matrix), which may or may not be symmetric
#         If A is not provided as a torch tensor, then an internal type conversion will be attempted
#         A similar type conversion is attempted for all inputs to GraphBuilder, but such operations will break the computational graph
#         Internally, the adjacency matrix is stored as neighbor_list representing a directed graph
#         If N is provided, a fully-connected graph with N nodes is initilized (with N*(N-1) directed edges)
        
#         ***** DYNAMIC GRAPHS *******************************************************************************************************
#         Unlike regular Graphs, dynamic Graphs (initialized with dynamic=True) can have their neighbor lists changed without losing
#         any edge feature data. That is, edges that are removed from the Graph still have their feature vectors stored internally.
#         ****************************************************************************************************************************

#         This initialization function defines the following graph fields:
#             (1) self.N                                      Number of nodes
#             (2) self.num_edges                              Number of edges (as inferred from the adjacency matrix)
#             (3) self.neighbor_list                          Neighbor list
#             (4) self.d                                      Degree vector (specifically, outdegree)
#             (5) self.dynamic                                Whether the graph is dynamic (default is False)

#         All other graph fields are left undefined. The function used to initially define each field is given in parentheses.
#             (6)  self.r                                     Node feature matrix (SetNodeLabels) 
#             (7)  self.Fr                                    Node feature vector length (SetNodeLabels)
#             (8)  self.e                                     Edge feature matrix (SetEdgeLabels)
#             (9)  self.e_full                                Full edge feature matrix for dynamic graphs (SetEdgeLabels)
#             (10) self.Fe                                    Edge feature vector length (SetEdgeLabels)
#             (11) self.edges_labeled_with_distances          Whether edges are labeled with their distances (SetEdgeLabels)
#             (12) self.y                                     Graph label (SetGraphLabel)
#             (13) self.xyz                                   3D coordinates of all nodes (SetXYZ)
#             (14) self.cutoff                                Cutoff distance (UpdateConnectivity)

#         Other args:
#             pbc (list or array): list of indices to include periodic boundary conditions on the graph.
#         """

#         if (N is not None and A is not None) or (N is None and A is None):
#             raise Exception("You must provide either N or A, but not both.")
#         else: 
#             if N is not None:
#                 self.N = N
#                 pairs = torch.LongTensor(-np.eye(self.N)+1).nonzero()
#             else:
#                 try:
#                     A.dtype
#                     passed_type_check = (A.dtype is torch.long)
#                     if passed_type_check == False: raise Exception()
#                 except:
#                     try:
#                         A = torch.LongTensor(A)
#                         #print("WARNING: Although A was successfully converted into a torch.LongTensor, this operation may have broken a computational graph (if there was one).")
#                     except:
#                         raise Exception("A must be a torch.LongTensor. An attempted type conversion failed.")
#                 self.N = A.shape[0]
#                 pairs = A.nonzero()
#             edge_indices = torch.zeros(self.N, self.N).to(torch.long)
#             off_diag_indices = torch.LongTensor(-np.eye(self.N)+1).nonzero()
#             edge_indices[off_diag_indices[:,0], off_diag_indices[:,1]] = torch.arange(self.N*(self.N-1))
#             edge_indices = edge_indices[pairs[:,0], pairs[:,1]].view(-1,1)
#             self.neighbor_list = torch.cat([edge_indices, pairs], dim=1)
#             self.num_edges = len(self.neighbor_list)
#             A = torch.zeros(self.N, self.N).to(torch.long)
#             A[self.neighbor_list[:,1], self.neighbor_list[:,2]] = 1
#             self.d = A.sum(1)
#         self.dynamic = dynamic
#         self.topology = None

#         if pbc is not None:
#             assert len(pbc) == self.N, \
#                 'pbc must have the same length of the number of atoms'

#             self.pbc = torch.LongTensor(pbc)
#         else:
#             self.pbc = torch.LongTensor(range(self.N))

#     def SetTopology(self, topology):
#         self.topology = topology

#     def SetNodeLabels(self, r, force=False):
#         """
#         r is the node feature matrix as a 2D torch.FloatTensor with shape (N, Fr), where Fr is the length of node feature vectors (the same for all nodes)
#         r[i] is the label (a.k.a. feature vector) of the ith node
#         r[i][j] is the jth feature of the ith node
#         This function defines the following graph fields: self.r, self.Fr
#         """
#         try:
#             self.r
#             already_exists = True
#         except:
#             already_exists = False
#         if not (already_exists == False or force == True):
#             raise Exception("You have already set the node labels. To change them, use force=True when calling SetNodeLabels().")
#         try:
#             r.dtype
#             passed_type_check = (r.dtype is torch.float)
#             if passed_type_check == False: raise Exception()
#         except:
#             try:
#                 r = torch.FloatTensor(r)
#                 #print("WARNING: Although r was successfully converted into a torch.FloatTensor, this operation may have broken a computational graph (if there was one).")
#             except:
#                 raise Exception("r must be a torch.FloatTensor. An attempted type conversion failed.")
#         if r.shape[0] != self.N:
#             raise Exception("The first dimension of r is inconsistent with the number of nodes in the graph.")
#         if len(r.shape) == 1:
#             r = r.view(-1,1)

#         # the reindexing ensures periodic boundary conditions are respected
#         self.r = r[self.pbc]

#         self.Fr = self.r.shape[1]

#     def SetEdgeLabels(self, e=None, force=False):
#         """
#         If provided, the edge feature matrix, e, must be a 2D torch.FloatTensor whose first dimension is either: 
#             (1) num_edges, in which case it contains labels for only the edges present in the graph as determined from the neighbor list
#             (2) N(N-1), in which case it contains labels for all possible edges in the graph (as if it were fully connected, even if it is not)
#         e[i] is the label (a.k.a. feature vector) of the ith edge
#         e[i][j] is the jth feature of the ith edge
#         If not provided, e will be saved as None
#         This function defines the following graph fields: self.e, self.Fe, self.edges_labeled_with_distances, and (for dynamic graphs) self.e_full
#         """
#         try:
#             self.e
#             already_exists = True
#         except:
#             already_exists = False
#         if not (already_exists == False or force == True):
#             raise Exception("You have already set the edge labels. To change them, use force=True when calling SetEdgeLabels().")
#         if e is not None:
#             try:
#                 e.dtype
#                 passed_type_check = (e.dtype is torch.float)
#                 if passed_type_check == False: raise Exception()
#             except:
#                 try:
#                     e = torch.FloatTensor(e)
#                     #print("WARNING: Although e was successfully converted into a torch.FloatTensor, this operation may have broken a computational graph (if there was one).")
#                 except:
#                     raise Exception("e must be a torch.FloatTensor. An attempted type conversion failed.")
#             if len(e.shape) == 1:
#                 e = e.view(-1,1)
#             if (self.dynamic == True) and (self.e.shape[0] != self.N*(self.N-1)):
#                 raise Exception("For graphs with dynamic=True, e must contain a feature vector for all N*(N-1) nodes.")
#             if e.shape[0] == int(self.N*(self.N-1)):
#                 self.e = e[self.neighbor_list[:,0]]
#             elif e.shape[0] == self.num_edges:
#                 self.e = e
#             else:
#                 raise Exception("The first dimension of e is inconsistent with the provided adjacency matrix and your decision to use dynamic=True.")
#             self.Fe = e.shape[1]
#         else:
#             self.e = None
#             self.Fe = 0
#         if self.dynamic:
#             self.e_full = e
#         self.edges_labeled_with_distances = False

#     def SetGraphLabel(self, y, force=False):
#         """
#         y is a torch.FloatTensor of arbitrary shape that labels this graph
#         This function defines the following graph fields: self.y
#         """
#         try:
#             self.y
#             already_exists = True
#         except:
#             already_exists = False
#         if not (already_exists == False or force == True):
#             raise Exception("You have already set the graph label. To change it, use force=True when calling SetGraphLabel().")
#         try:
#             y.dtype
#             passed_type_check = (y.dtype is torch.float)
#             if passed_type_check == False: raise Exception()
#         except:
#             try:
#                 y = torch.FloatTensor(y)
#                 #print("WARNING: Although y was successfully converted into a torch.FloatTensor, this operation may have broken a computational graph (if there was one).")
#             except:
#                 raise Exception("y must be a torch.FloatTensor. An attempted type conversion failed.")
#         self.y = y

#     def SetXYZ(self, xyz, force=False):
#         """
#         Sets the 3D coordinates of the nodes using xyz, a 2D torch.FloatTensor with shape (N, 3)
#         xyz[i][j] is the jth coordinate (j=0,1,2 for x,y,z) of the ith node
#         This function defines the following graph fields: self.xyz
#         """
#         try:
#             self.xyz
#             already_exists = True
#         except:
#             already_exists = False
#         if not (already_exists == False or force == True):
#             raise Exception("You have already set the node positions. To change them, use force=True when calling SetXYZ().")
#         """
#         try:
#             xyz.dtype
#             passed_type_check = (xyz.dtype is torch.float)
#             if passed_type_check == False: raise Exception()
#         except:        
#             try:
#                 #import IPython; IPython.embed()
#                 xyz = torch.tensor(xyz.tolist(), requires_grad=True)
#                 #xyz = torch.zeros(xyz.shape)
#                 #torch.nn.Parameter(tensor(xyz, requires_grad=False).to(torch.float))
#                 #xyz.requires_grad = True
#                 #print("WARNING: Although xyz was successfully converted into a torch.FloatTensor, this operation may have broken a computational graph (if there was one).")
#             except:
#                 raise Exception("xyz must be a torch.FloatTensor. An attempted type conversion failed.")
#         """
#         if xyz.shape[0] != self.N:
#             raise Exception("The dimensions of xyz are inconsistent with the number of nodes in the graph.")
#         if xyz.shape[1] != 3:
#             raise Exception("The second dimension of xyz should be 3 (x, y, and z positions).")
#         self.xyz = torch.tensor(xyz.tolist(), requires_grad=True, dtype=torch.float)

#     def LabelEdgesWithDistances(self):
#         """
#         Prepends the distance to each edge feature vector. You must use SetXYZ() to set node positions before calling this function.
#         """
#         try:
#             self.e
#             if self.edges_labeled_with_distances == True:
#                 raise Exception("Edges have already been labeled with distances.")
#         except:
#             self.SetEdgeLabels(e=None, force=True)
#         try:
#             self.xyz
#         except:
#             raise Exception("You must first set the node positions using SetXYZ() if you want to label edges with their distances.")
#         R = self.xyz.expand(self.N, self.N, 3) - self.xyz.expand(self.N, self.N, 3).transpose(0, 1)
#         D = R.pow(2).sum(dim=2).sqrt()
#         if self.e is None:
#             if self.dynamic:
#                 off_diag_indices = torch.LongTensor(-np.eye(self.N)+1).nonzero()
#                 self.e_full = D[off_diag_indices[:,0], off_diag_indices[:,1]].view(-1,1)
#             self.e = D[self.neighbor_list[:,1], self.neighbor_list[:,2]].view(-1,1)
#         else:
#             if self.dynamic:
#                 self.e_full = torch.cat([D[off_diag_indices[:,0], off_diag_indices[:,1]].view(-1,1), self.e_full], dim=1)
#             self.e = torch.cat([D[self.neighbor_list[:,1], self.neighbor_list[:,2]].view(-1,1), self.e], dim=1)
#         self.Fe += 1
#         self.edges_labeled_with_distances = True

#     def GaussianExpandDistances(self, offset, widths):
#         if self.edges_labeled_with_distances == False:
#             self.LabelEdgesWithDistances()
#         def gaussian_smearing(distances, offset, widths, centered=False, graph=False):
#             if centered == False:
#                 coeff = -0.5 / torch.pow(widths, 2)
#                 if graph is not True:
#                     diff = distances[:, :, :, None] - offset[None, None, None, :]
#                 else:
#                     diff = distances - offset
#             else:
#                 coeff = -0.5 / torch.pow(offset, 2)
#                 if graph is not True:
#                     diff = distances[:, :, :, None]
#                 else:
#                     diff = distances
#             gauss = torch.exp(coeff * torch.pow(diff, 2))
#             return(gauss)
#         dist = self.e[:,0].view(1,-1,1)
#         gauss_dist = gaussian_smearing(distances=dist, offset=offset, widths=widths)
#         self.e = torch.cat([gauss_dist[0,:,0,:], self.e[:,1:]], dim=1)
#         self.Fe = self.e.shape[1]

#     def UpdateConnectivity(self, A=None, cutoff=None, threshold=0):
#         """
#         Update the neighbor list either by providing a new adjacency matrix or by providing a cutoff distance.
#         If updating with a cutoff distance, the node positions must be set using SetXYZ(). 
#         """
#         if (cutoff is not None and A is not None) or (cutoff is None and A is None):
#             raise Exception("You must provide either A or a cutoff, but not both.")
#         if (self.num_edges != self.N*(self.N-1)) and (self.dynamic == False):
#             raise Exception("A cutoff can only be applied to fully connected graphs or graphs with dynamic=True.")
#         try:
#             self.e
#         except:
#             self.e = None
#             if self.dynamic:
#                 self.e_full = None
#         if A is None:
#             try:
#                 self.xyz
#             except:
#                 raise Exception("You must first set the node positions using SetXYZ() if you want to label edges with their distances.")
#             R = self.xyz.expand(self.N, self.N, 3) - self.xyz.expand(self.N, self.N, 3).transpose(0, 1)
#             D = R.pow(2).sum(dim=2).sqrt()
#             self.cutoff = cutoff
#             self.threshold = threshold
#             mask = (self.threshold <= D) & (D <= self.cutoff)
#             mask[np.diag_indices(self.N)] = 0
#             pairs = mask.nonzero()
#         else:
#             try:
#                 A.dtype
#                 passed_type_check = (A.dtype is torch.long)
#                 if passed_type_check == False: raise Exception()
#             except:
#                 try:
#                     A = torch.LongTensor(A)
#                     #print("WARNING: Although A was successfully converted into a torch.LongTensor, this operation may have broken a computational graph (if there was one).")
#                 except:
#                     raise Exception("A must be a torch.LongTensor. An attempted type conversion failed.")
#             if A.shape[0] != self.N:
#                 raise Exception("This adjacency matrix does not have the correct dimensions.")
#             pairs = A.nonzero()
#         edge_indices = torch.zeros(self.N, self.N).to(torch.long)
#         off_diag_indices = torch.LongTensor(-np.eye(self.N)+1).nonzero()
#         edge_indices[off_diag_indices[:,0], off_diag_indices[:,1]] = torch.arange(self.N*(self.N-1))
#         edge_indices = edge_indices[pairs[:,0], pairs[:,1]].view(-1,1)
#         self.neighbor_list = torch.cat([edge_indices, pairs], dim=1)
#         self.num_edges = len(self.neighbor_list)
#         if self.e is not None:
#             if self.dynamic:
#                 self.e = self.e_full[self.neighbor_list[:,0]]
#             else:
#                 self.e = self.e[self.neighbor_list[:,0]]
#         A = torch.zeros(self.N, self.N).to(torch.long)
#         A[self.neighbor_list[:,1], self.neighbor_list[:,2]] = 1
#         self.d = A.sum(1)
#     def OneHotNodes(self, indices=None, sets=None):
#         graph_dataset = GraphDataset(dynamic=False)
#         graph_dataset.AddGraph(self)
#         graph_dataset.CreateBatches(batch_size=1)
#         graph_dataset.OneHotNodes(indices=indices, sets=sets)
#         self.r = graph_dataset.batches[0].data['r']
#         self.Fr = graph_dataset.Fr







# def get_atomic_graph(smiles=None, add_hydrogens=False, atomic_nums=None, A=None, xyz=None, ATOM_CODES=None,
#     dim=2, return_rdkit_mol=False):
#     if smiles is None:
#         if (atomic_nums is None) and (xyz is None):
#             raise Exception("You must provide either smiles or (atomic_nums and xyz), but not both.")
#     else:
#         if atomic_nums is not None:
#             raise Exception("You must provide either smiles or (atomic_nums and xyz), but not both.")
#     #if dim != 2:
#     #    raise Exception("Only 2D molecular graphs are currently supported.")
#     if smiles is not None:
#         rdkit_mol = Chem.MolFromSmiles(smiles)
#         if add_hydrogens:
#             rdkit_mol = Chem.AddHs(rdkit_mol)
#         atomic_nums = [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
#         A = Chem.GetAdjacencyMatrix(rdkit_mol)
#         N = rdkit_mol.GetNumAtoms()
#         Q = rdkit_mol.GetNumBonds()
#         if dim == 2:
#             Chem.Compute2DCoords(rdkit_mol)
#             mol_block = Chem.MolToMolBlock(rdkit_mol)
#             mol_block = mol_block.split('\n')
#             mol_block = mol_block[4:4+N]
#             mol_block = pd.DataFrame(mol_block)
#             mol_block = mol_block[0].str.split(expand=True)
#             mol_block = mol_block[[0,1,3]]
#             mol_block = mol_block.rename(index=int, columns={0: "x", 1: "y", 3: "element"})
#             mol_block['x'] = mol_block['x'].astype(float)
#             mol_block['y'] = mol_block['y'].astype(float)
#             mol_block['atomic_num'] = atomic_nums
#             xyz = np.zeros((N,3))
#             xyz[:,:2] += np.array(mol_block.iloc[:,0:2])
#         elif dim == 3:
#             conformerID = Chem.EmbedMolecule(rdkit_mol, Chem.ETKDG())
#             Chem.UFFOptimizeMolecule(rdkit_mol)
#             rdkit_conformer = rdkit_mol.GetConformer(conformerID)
#             coords = rdkit_conformer.GetPositions().tolist()
#             xyz = []
#             for i in range(rdkit_mol.GetNumAtoms()):
#                 row = []
#                 row.append(float(coords[i][0]))
#                 row.append(float(coords[i][1]))
#                 row.append(float(coords[i][2]))
#                 xyz.append(row)
#             xyz = np.array(xyz)
#         else:
#             raise Exception("dim must be 2 or 3.")
#         adjmat = np.zeros((N,N))
#         adjmat[np.triu_indices(N, k=1)] += A[np.triu_indices(N, k=1)]
#         bond_list = adjmat.nonzero()
#         a = np.stack([bond_list[0], bond_list[1]], axis=1)
#     else:
#         N = len(atomic_nums)
#     if ATOM_CODES is None:
#         r = np.array(atomic_nums).astype(int)
#     else:
#             z = pd.DataFrame()
#             z['atomic_num'] = atomic_nums
#             z['one_hot'] = z['atomic_num'].map(lambda a: ATOM_CODES.loc[a]['one_hot'])
#             r = np.array(list(z['one_hot'])).astype(int)
#     if smiles is None:
#         if A is None:
#             ligand_graph = graphbuilder.Graph(N=N) #, dynamic=False)
#         else:
#             # A = np.array(A)
#             ligand_graph = graphbuilder.Graph(A=A) #, dynamic=False)
#     else:
#         ligand_graph = graphbuilder.Graph(A=A) # dynamic=False)
#     ligand_graph.SetNodeLabels(r=r)
#     ligand_graph.SetXYZ(xyz=xyz)
#     ligand_graph.LabelEdgesWithDistances()
#     ligand_graph.SetGraphLabel(y=[0])
#     if return_rdkit_mol:
#         return(ligand_graph, rdkit_mol)
#     else:
#         return(ligand_graph)








# geom_data.A = torch.tensor(geom_data.A).to(torch.long)
# geom_data.atomic_nums = torch.tensor(geom_data.xyz)[:,0].to(torch.long)
# geom_data.xyz = torch.tensor(geom_data.xyz)[:,-3:]
# geom_data.charges = torch.tensor(geom_data.charges)[:,[0]]

# geom_graph = binding.utils.get_atomic_graph(
#     atomic_nums=geom_data.atomic_nums,
#     A=geom_data.A,
#     xyz=geom_data.xyz)




# topology = Topology(graph=geom_graph, use_1_4_pairs=job_details.use_1_4_pairs)
# geom_graph.SetTopology(topology=topology)
# # probably don't need this:
# geom_graph = assign_true_params(geom_graph, geom_data)


# # need to figure this out:
# geom_graph.SetEdgeLabels(e=torch.zeros_like(geom_graph.e), force=True)
# geom_graph.SetGraphLabel(y=torch.zeros(1,1), force=True)



# ################################################################################
# ################################################################################
# ################################################################################

# class GraphDataset():

#     def __init__(self, dynamic=False):
#         """
#         ADD DOCUMENTATION
#         """
#         self.container_batch = GraphBatch()
#         self.dynamic = dynamic

#     def AddGraph(self, graph):
#         self.container_batch.AddGraph(graph)

#     def CreateBatches(self, batch_size, force=False, finalize=True, show_output=False):
#         """
#         ADD DOCUMENTATION
#         """
#         try:
#             self.batches
#             already_batched = True
#         except:
#             already_batched = False
#         if already_batched == True:
#             if force == False:
#                 raise Exception("This GraphDataset has already been batched. To re-batch it, use force=True when calling CreateBatches().")
#             else:
#                 del(self.batches)
#                 del(self.batch_size)
#                 del(self.dataset_size)
#                 del(self.num_batches)
#                 del(self.Fr)
#                 del(self.Fe)
#                 del(self.Y)

#         self.batch_size = batch_size
#         self.dataset_size = len(self.container_batch.graphs)

#         indices = [i for i in range(self.dataset_size)]
#         batches_of_indices = [indices[i:i+self.batch_size] for i in range(0, self.dataset_size, self.batch_size)]
#         self.num_batches = len(batches_of_indices)

#         self.batches = []
#         for batch in range(self.num_batches):
#             self.batches.append(GraphBatch())
#             B = len(batches_of_indices[batch])
#             for i in batches_of_indices[batch]:
#                 self.batches[-1].AddGraph(self.container_batch.graphs[i])
#             if finalize: self.batches[-1].FinalizeBatch(dynamic=self.dynamic)
#             if show_output:
#                 print("Finished finalizing batch {} of {}.".format(batch+1, self.num_batches))

#         self.Fr = self.batches[0].Fr
#         self.Fe = self.batches[0].Fe
#         self.Y = self.batches[0].Y

#     def OneHotNodes(self, indices=None, sets=None):
#         """
#         Apply a one-hot transformation to the node features in self.data['r'] specified by the given indices.
#         If no indices are given, then each column of self.data['r'] will be one-hot encoded.
#         """
#         try:
#             self.batches
#         except:
#             raise Exception("Node feature transformations can only be applied to batched GraphDatasets.")
#         if indices is not None:
#             if type(indices) != list:
#                 raise Exception("indices must be a list of integers.")
#         #torch.set_grad_enabled(False)
#         for b in range(self.num_batches):
#             r = self.batches[b].data['r']
#             r_list = list(r.split([1 for j in range(r.shape[1])], dim=1))
#             new_ri_list = []
#             for i in range(self.Fr):
#                 if i in indices:
#                     if sets[indices.index(i)] is not None:
#                         ordered_unique_values = sets[indices.index(i)]
#                         df = pd.DataFrame()
#                         ri = np.array(r_list[i].view(-1).tolist())
#                         df['ri'] = ri
#                         df['index'] = df['ri'].map(lambda x: ordered_unique_values.index(x))
#                         new_ri = torch.zeros(len(ri), len(ordered_unique_values))
#                         new_ri[[k for k in range(len(new_ri))], list(df['index'])] += 1
#                     else:
#                         ri = r_list[i].view(-1).unique(sorted=True, return_inverse=True)[1]
#                         num_unique_values = len(ri.unique())
#                         new_ri = torch.zeros(len(ri), num_unique_values)
#                         new_ri[[k for k in range(len(new_ri))], ri] += 1
#                 else:
#                     new_ri = r_list[i].view(-1,1)
#                 new_ri_list.append(new_ri)
#             self.batches[b].data['r'] = torch.cat(new_ri_list, dim=1)
#             self.batches[b].Fr = self.batches[b].data['r'].shape[1]
#         self.Fr = self.batches[0].Fr

#     def GaussianExpandDistances(self, offset, widths, force=False):
#         try:
#             self.batches
#             if force is False:
#                 if self.batches[0].edges_labeled_with_distances == False:
#                     raise Exception("Edges must first be labeled with distance.")
#         except:
#             raise Exception("GaussianExpandDistances() can only be called on a batched dataset for which edges_labeled_with_distance = True.")
#         for b in range(self.num_batches):
#             dist = self.batches[b].data['e'][:,0].view(1,-1,1)
#             coeff = -0.5 / torch.pow(widths, 2)
#             diff = dist[:, :, :, None] - offset[None, None, None, :]
#             gauss_dist =  torch.exp(coeff * torch.pow(diff, 2)) #torch.exp(dist.unsqueeze(3))
#             self.batches[b].data['e'] = torch.cat(
#                 [
#                     gauss_dist[0,:,0,:],
#                     self.batches[b].data['e'][:,1:]
#                 ],
#                 dim=1
#             )
#             self.batches[b].Fe = self.batches[b].data['e'].shape[1]
#         self.Fe = self.batches[0].Fe
                                        
# ################################################################################
# ################################################################################
# ################################################################################

# def dataset_from_frames(job_details, frames, geom_ids):
#     if job_details.subtract_reference_energy:
#         frames = subtract_reference_energy(frames)
#     frames = optionally_normalize_property(job_details, frames)


#     # unique atomic numbers (e.g., [1, 6, 7])
#     z_set = torch.cat(frames.atomic_nums).unique().tolist()

#     graph_datasets = {}
#     for split in ['train', 'test']:
#         graph_datasets[split] = graphbuilder.GraphDataset(dynamic=False)
#     for counter, geom_id in enumerate(frames.geom_id):
#         if geom_id in geom_ids['train']:
#             split = 'train'
#         elif geom_id in geom_ids['test']:
#             split = 'test'
#         else:
#             raise Exception("There was a frame for a Geom that is not in the train or test sets.")
#         y = frames[job_details.target_property][counter]
#         if job_details.target_property == 'E':
#             y = y.view(-1,1)
#         geom_graph = frames.graph[counter]
#         if job_details.force_distinct_atom_types:
#             geom_graph.SetNodeLabels(r=torch.eye(geom_graph.N).to(torch.float), force=True)
#         elif job_details.force_graph_distinct_atom_types:
#             geom_graph.OneHotNodes(indices=[0], sets=[z_set])
#             r = copy.deepcopy(geom_graph.r)
#             a = geom_graph.neighbor_list[:,-2:]

#             ##### this looks important


#             for k in range(job_details.num_convolutions):
#                 messages = list(torch.split(r[a[:,1]], geom_graph.d.tolist()))
#                 messages = [messages[n].sum(0) for n in range(geom_graph.N)]
#                 messages = torch.stack(messages)
#                 node_and_message = torch.cat([r, messages], dim=1)
#                 eye_indices = np.unique(node_and_message, axis=0, return_inverse=True)[1]
#                 r = torch.eye(int(eye_indices.max())+1)[eye_indices]
#             geom_graph.SetNodeLabels(r=r, force=True)
#         else:
#             geom_graph.OneHotNodes(indices=[0], sets=[z_set])
#         geom_graph.SetGraphLabel(y=y, force=True)
#         graph_datasets[split].AddGraph(geom_graph)
#     for split in ['train', 'test']:
#         graph_datasets[split].CreateBatches(batch_size=job_details.batch_size)
#     return(graph_datasets, frames)


# graph_datasets, frames = dataset_from_frames(job_details, frames, geom_ids)


