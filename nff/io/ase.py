import os
import numpy as np
import torch

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes

import nff.utils.constants as const
from nff.nn.utils import torch_nbr_list
from nff.utils.cuda import batch_to
from nff.data.sparse import sparsify_array


DEFAULT_CUTOFF = 5.0


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
        Atoms objects.
    """

    def __init__(
        self,
        *args,
        props={},
        cutoff=DEFAULT_CUTOFF,
        nbr_torch=False,
        **kwargs
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        self.props = props
        self.nbr_list = props.get('nbr_list', None)
        self.offsets = props.get('offsets', None)
        self.nbr_torch = nbr_torch
        self.num_atoms = props.get('num_atoms', len(self))
        self.cutoff = cutoff
        self.device = 0

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
            inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
            to be sent to the model.
        Returns:
            batch (dict): batch with the keys 'nxyz',
                'num_atoms', 'nbr_list' and 'offsets'
        """
        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()
        self.props['nbr_list'] = self.nbr_list
        self.props['offsets'] = self.offsets

        self.props['nxyz'] = torch.Tensor(self.get_nxyz())
        self.props['num_atoms'] = torch.LongTensor([len(self)])
 
        return self.props

    def update_nbr_list(self):
        """Update neighbor list and the periodic reindexing
            for the given Atoms object.

        Args:
            cutoff (float): maximum cutoff for which atoms are
                considered interacting.
        Returns:
            nbr_list (torch.LongTensor)
            offsets (torch.Tensor)
            nxyz (torch.Tensor)
        """

        if self.nbr_torch:
            edge_from, edge_to, offsets = torch_nbr_list(self, self.cutoff, device=self.device)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        else:
            edge_from, edge_to, offsets = neighbor_list('ijS', self, self.cutoff) 
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            offsets = torch.Tensor(offsets)[nbr_list[:, 1] > nbr_list[:, 0]].detach().cpu().numpy()
            nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

        # torch.sparse has no storage yet.
        #offsets = offsets.dot(self.get_cell())
        offsets = sparsify_array(offsets.dot(self.get_cell()))

        self.nbr_list = nbr_list
        self.offsets = offsets

        return nbr_list, offsets

    def batch_properties():
        pass

    def batch_kinetic_energy():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls, atoms):
        return cls(
            atoms,
            positions=atoms.positions,
            numbers=atoms.numbers,
            props={},
        )


class BulkPhaseMaterials(Atoms):
    """Class to deal with the Neural Force Field and batch molecules together
    in a box for handling boxphase. 
    """

    def __init__(
        self,
        *args,
        props={},
        cutoff=DEFAULT_CUTOFF,
        nbr_torch=False,
        device='cpu',
        **kwargs
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        self.props = props
        self.nbr_list = self.props.get('nbr_list', None)
        self.offsets = self.props.get('offsets', None)
        self.num_atoms = self.props.get('num_atoms', None)
        self.cutoff = cutoff
        self.nbr_torch = nbr_torch
        self.device = device 

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
            inside the unit cell of the system.

        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
            to be sent to the model.

        Returns:
            batch (dict): batch with the keys 'nxyz',
                'num_atoms', 'nbr_list' and 'offsets'
        """
        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()
            self.props['nbr_list'] = self.nbr_list
            self.props['atoms_nbr_list'] = self.atoms_nbr_list
            self.props['offsets'] = self.offsets

        self.props['nbr_list'] = self.nbr_list
        self.props['atoms_nbr_list'] = self.atoms_nbr_list
        self.props['offsets'] = self.offsets
        self.props['nxyz'] = torch.Tensor(self.get_nxyz())

        return self.props

    def update_system_nbr_list(self, cutoff, exclude_atoms_nbr_list=True):
        """Update undirected neighbor list and the periodic reindexing
            for the given Atoms object.ß

        Args:
            cutoff (float): maximum cutoff for which atoms are
                considered interacting.

        Returns:
            nbr_list (torch.LongTensor)
            offsets (torch.Tensor)
            nxyz (torch.Tensor)
        """
        if self.nbr_torch:
            edge_from, edge_to, offsets = torch_nbr_list(self, self.cutoff, device=self.device)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        else:
            edge_from, edge_to, offsets = neighbor_list('ijS', self, self.cutoff) 
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            offsets = torch.Tensor(offsets)[nbr_list[:, 1] > nbr_list[:, 0]].numpy()
            nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

        if exclude_atoms_nbr_list:
            offsets_mat = torch.zeros(self.get_number_of_atoms(),
                                      self.get_number_of_atoms(), 3)
            nbr_list_mat = torch.zeros(self.get_number_of_atoms(), 
                                       self.get_number_of_atoms()).to(torch.long)
            atom_nbr_list_mat = torch.zeros(self.get_number_of_atoms(), 
                                            self.get_number_of_atoms()).to(torch.long)
            
            offsets_mat[nbr_list[:, 0], nbr_list[:, 1]] = offsets
            nbr_list_mat[nbr_list[:, 0], nbr_list[:, 1]] = 1
            atom_nbr_list_mat[self.atoms_nbr_list[:, 0], self.atoms_nbr_list[:, 1]] = 1

            nbr_list_mat = nbr_list_mat - atom_nbr_list_mat
            nbr_list = nbr_list_mat.nonzero()
            offsets = offsets_mat[nbr_list[:,0], nbr_list[:,1], :]

        self.nbr_list = nbr_list
        self.offsets = sparsify_array(offsets.dot(self.get_cell()))
        
    def get_list_atoms(self):

        mol_split_idx = self.props['num_subgraphs'].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            Atoms_list.append(Atoms(Z[i].tolist(),
                                    molecule_xyz.numpy(),
                                    cell=self.cell,
                                    pbc=self.pbc))

        return Atoms_list

    def update_atoms_nbr_list(self, cutoff):

        Atoms_list = self.get_list_atoms()

        intra_nbr_list = []
        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to = neighbor_list('ij', atoms, cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]
            intra_nbr_list.append(
                self.props['num_subgraphs'][: i].sum() + nbr_list)

        intra_nbr_list = torch.cat(intra_nbr_list)
        self.atoms_nbr_list = intra_nbr_list
    
    def update_nbr_list(self):
        self.update_atoms_nbr_list(self.props['atoms_cutoff'])
        self.update_system_nbr_list(self.props['system_cutoff'])


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        device='cpu',
        **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
            model (TYPE): Description
            device (str): device on which the calculations will be performed 
            **kwargs: Description
            model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
        self,
        atoms=None,
        properties=['energy', 'forces'],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
            atoms (AtomsBatch): custom Atoms subclass that contains implementation
                of neighbor lists, batching and so on. Avoids the use of the Dataset
                to calculate using the models created.
            properties (list of str): 'energy', 'forces' or both
            system_changes (default from ase)
        """

        Calculator.calculate(self, atoms, properties, system_changes)

        # run model
        #atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        batch['energy'] = []
        if 'forces' in properties:
            batch['energy_grad'] = []

        prediction = self.model(batch)

        # change energy and force to numpy array
        energy = prediction['energy'].detach().cpu(
        ).numpy() * (1 / const.EV_TO_KCAL_MOL)
        energy_grad = prediction['energy_grad'].detach(
        ).cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)

        self.results = {
            'energy': energy.reshape(-1),
        }

        if 'forces' in properties:
            self.results['forces'] = -energy_grad.reshape(-1, 3)

    @classmethod
    def from_file(
        cls,
        model_path,
        device='cuda',
        **kwargs
    ):
        model = nff.train.builders.models.load_model(model_path)
        return cls(model, device, **kwargs)
