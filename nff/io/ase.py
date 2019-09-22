import os 
import numpy as np
import torch

from ase import Atoms
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes

from nff.utils.scatter import compute_grad
import nff.utils.constants as const
from nff.train.builders import load_model


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
        self.ase_nl = NeighborList(
            [cutoff / 2] * len(self),
            bothways=True,
            self_interaction=False
        )
        self.nbr_list = props.get('nbr_list', None)
        self.pbc_index = props.get('pbc', None)
        self.props = props
        self.nxyz = self.get_nxyz()

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
                'num_atoms', 'nbr_list' and 'pbc'
        """
        if self.nbr_list is None or self.pbc_index is None:
            self.update_nbr_list()

        # wwj: what if my model has other inputs? 
        # batch = {
        #     'nxyz': torch.Tensor(self.nxyz),
        #     'num_atoms': torch.LongTensor([len(self)]),
        #     'nbr_list': self.nbr_list,
        #     'pbc': self.pbc_index
        # }

        self.props['nxyz'] = torch.Tensor(self.get_nxyz())

        return self.props

    def update_nbr_list(self, cutoff):
        """Update neighbor list and the periodic reindexing
            for the given Atoms object.
        
        Args:
            cutoff (float): maximum cutoff for which atoms are
                considered interacting.

        Returns:
            nbr_list (torch.LongTensor)
            pbc_index (torch.LongTensor)
        """

        self.ase_nl.update(self)
        indices = np.concatenate(self.ase_nl.nl.neighbors, axis=0).reshape(-1, 1)
        offsets = np.concatenate(self.ase_nl.nl.displacements, axis=0)
        neighbors = np.concatenate([indices, offsets], axis=1) 
        atoms_idx = np.unique(neighbors, axis=0)

        pair_left = np.concatenate([
            [atom] * len(nbrs)
            for atom, nbrs in enumerate(self.ase_nl.nl.neighbors)      
        ], axis=0)

        pair_right = np.where(
            np.bitwise_and.reduce(
                neighbors[:, None] == atoms_idx[None, :],
                axis=-1
            )
        )[1]

        nbr_list = np.stack([pair_left, pair_right], axis=1)
        pbc_index = atoms_idx[:, 0]
        xyz = self.get_positions()[atoms_idx[:, 0]] + \
                np.dot(atoms_idx[:, 1:], self.get_cell())
        nxyz = np.concatenate([
            self.get_atomic_numbers()[atoms_idx[:, 0]].reshape(-1, 1),
            xyz
        ], axis=1)
        
        nbr_list = torch.LongTensor(nbr_list)
        pbc_index = torch.LongTensor(pbc_index)
        nxyz = torch.Tensor(nxyz)

        self.nbr_list = nbr_list
        self.pbc_index = pbc_index
        self.nxyz = nxyz

    def batch_properties():
        pass 

    def batch_kinetic_energy(self):
        pass
    
    def batch_virial(self):
        pass


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        props,
        device='cpu',
        **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py
        
        Args:
            model (TYPE): Description
            device (str, optional): Description
            model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.props = props

    def to(self, device):
        self.device = device

    def calculate(
        self,
        atoms=None,
        properties=['energy'],
        device=0,
        system_changes=all_changes,
    ):
        
        Calculator.calculate(self, atoms, properties, system_changes)

        # run model 
        batch = atoms.get_batch()

        prediction = self.model(self.props)
        
        # change energy and force to numpy array 
        energy = prediction['energy'].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        grad = prediction['energy_grad'].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        
        self.results = {
            'energy': energy.reshape(-1),
            'forces': -grad.reshape(-1, 3)
        }

    @classmethod
    def from_file(
        cls,
        model_path,
        device='cuda',
        **kwargs
    ):
        model = load_model(model_path)
        return cls(model, device, **kwargs)

