import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes

import nff.utils.constants as const
from nff.train import load_model, evaluate


DEFAULT_CUTOFF = 5.0


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
        Atoms objects.
    """

    def __init__(
        self,
        *args,
        nbr_list=None,
        offsets=None,
        cutoff=DEFAULT_CUTOFF,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nbr_list = nbr_list
        self.offsets = offsets
        self.nxyz = self.get_nxyz()
        self.cutoff = cutoff

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

        batch = {
            'nxyz': torch.Tensor(self.nxyz),
            'num_atoms': torch.LongTensor([len(self)]),
            'nbr_list': self.nbr_list,
            'offsets': self.offsets
        }
        return batch

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

        edge_from, edge_to, offsets = neighbor_list('ijS', self, self.cutoff) 
        nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        offsets = torch.Tensor(offsets.dot(self.get_cell()))

        xyz = self.get_positions()
        nxyz = np.concatenate([self.get_atomic_numbers().reshape(-1, 1), xyz], axis=1)
        nxyz = torch.Tensor(nxyz)

        self.nbr_list = nbr_list
        self.offsets = offsets
        self.nxyz = nxyz

        return nbr_list, offsets, nxyz


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        device='cuda',
        **kwargs
    ):
        """Creates a NeuralFF calculator.

        Args:
            model (one of nff.nn.models)
            device (str): device on which the calculations will be performed 
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device

    def calculate(
        self,
        atoms=None,
        properties=['energy'],
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

        batch = atoms.get_batch()

        raise NotImplementedError

        # to compute the kinetic energies to this...
        #mass = atoms.get_masses()
        # vel = atoms.get_velocities()
        # vel = torch.Tensor(vel)
        # mass = torch.Tensor(mass)

        # print(atoms.get_kinetic_energy())
        # print(atoms.get_kinetic_energy().dtype)
        # print( (0.5 * (vel * 1e-10 * fs * 1e15).pow(2).sum(1) * (mass * 1.66053904e-27) * 6.241509e+18).sum())
        # print( (0.5 * (vel * 1e-10 * fs * 1e15).pow(2).sum(1) * (mass * 1.66053904e-27) * 6.241509e+18).sum().type())

        # rebtach based on the number of atoms


        # change energy and force to numpy array 
        #energy = energy.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        #forces = forces.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        
        self.results = {
            'energy': energy.reshape(-1),
            'forces': forces
        }

    def get_kinetic_energy(self):
        pass

    @classmethod
    def from_file(
        cls,
        model_path,
        bond_adj=None,
        bond_len=None,
        device='cuda',
        **kwargs
    ):
        model = load_model(model_path)
        return cls(model, bond_adj, bond_len, device, **kwargs)

