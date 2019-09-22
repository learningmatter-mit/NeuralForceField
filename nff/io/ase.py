import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import Atoms
from ase.neighborlist import neighbor_list
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
        nbr_list=None,
        pbc_index=None,
        cutoff=DEFAULT_CUTOFF,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nbr_list = nbr_list
        self.pbc_index = pbc_index
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
                'num_atoms', 'nbr_list' and 'pbc'
        """
        if self.nbr_list is None or self.pbc_index is None:
            self.update_nbr_list()

        batch = {
            'nxyz': torch.Tensor(nxyz),
            'num_atoms': torch.LongTensor([len(self)]),
            'nbr_list': self.nbr_list,
            'pbc': self.pbc_index
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
            pbc_index (torch.LongTensor)
        """

        edge_from, edge_to, distances = neighbor_list('ijd', self, self.cutoff) 
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

        return 


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        bond_adj=None,
        bond_len=None,
        device='cuda',
        **kwargs
    ):
        """Creates a NeuralFF calculator.

        Args:
            model (one of nff.nn.models)
            device (str)
            bond_adj (? or None)
            bond_len (? or None)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.bond_adj = bond_adj
        self.bond_len = bond_len
        self.device = device

    def to(self, device):
        self.device = device

    def calculate(
        self,
        atoms=None,
        properties=['energy'],
        system_changes=all_changes,
    ):
        
        Calculator.calculate(self, atoms, properties, system_changes)

        # number of atoms 
        num_atoms = atoms.get_atomic_numbers().shape[0]

        # run model 
        atomic_numbers = atoms.get_atomic_numbers()#.reshape(1, -1, 1)
        xyz = atoms.get_positions()#.reshape(-1, num_atoms, 3)
        bond_adj = self.bond_adj
        bond_len = self.bond_len

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

        atomic_numbers = torch.LongTensor(atomic_numbers).to(self.device).reshape(-1, num_atoms)

        xyz = torch.Tensor(xyz).to(self.device).reshape(-1, num_atoms, 3)
        self.model.to(self.device)

        xyz.requires_grad = True

        energy = self.model(
            r=atomic_numbers,
            xyz=xyz,
            bond_adj=bond_adj,
            bond_len=bond_len
        )

        forces = -compute_grad(inputs=xyz, output=energy)

        kin_energy = self.get_kinetic_energy()

        # change energy and forces back 
        energy = energy.reshape(-1)
        forces = forces.reshape(-1, 3)
        
        # change energy and force to numpy array 
        energy = energy.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        forces = forces.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        
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

