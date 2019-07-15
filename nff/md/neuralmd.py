import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase.calculators.calculator import Calculator, all_changes

from nff.utils.scatter import compute_grad
import nff.utils.constants as const


class NeuralMD(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device, N_atom, bondAdj=None, bondlen=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.N_atom = N_atom
        # declare adjcency matrix 
        self.bondAdj = bondAdj
        self.bondlen = bondlen

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        
        Calculator.calculate(self, atoms, properties, system_changes)

        # number of atoms 
        #n_atom = atoms.get_atomic_numbers().shape[0]
        N_atom = self.N_atom
        # run model 
        node = atoms.get_atomic_numbers()#.reshape(1, -1, 1)
        xyz = atoms.get_positions()#.reshape(-1, N_atom, 3)
        bondAdj = self.bondAdj
        bondlen = self.bondlen

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

        node = Variable(torch.LongTensor(node).reshape(-1, N_atom)).cuda(self.device)
        xyz = Variable(torch.Tensor(xyz).reshape(-1, N_atom, 3)).cuda(self.device)
        xyz.requires_grad = True

        # predict energy and force
        if bondlen is not None and bondAdj is not None:
            U = self.model(r=node, xyz=xyz, bonda=bondAdj, bondlen=bondlen)
            f_pred = -compute_grad(inputs=xyz, output=U)
        else:
            U = self.model(r=node, xyz=xyz)
            f_pred = -compute_grad(inputs=xyz, output=U)

        # change energy and forces back 
        U = U.reshape(-1)
        f_pred = f_pred.reshape(-1, 3)
        
        # change energy and force to numpy array 
        energy = U.detach().cpu().numpy() * (1 / const.EV_TO_KCAL)
        forces = f_pred.detach().cpu().numpy() * (1 / const.EV_TO_KCAL)
        
        self.results = {
            'energy': energy.reshape(-1),
            'forces': forces
        }
