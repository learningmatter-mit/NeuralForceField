import os 
import numpy as np
import torch
from torch.autograd import Variable

import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import Atoms
from ase.units import Bohr, Rydberg, kJ, kB, fs, Hartree, mol, kcal
from ase.md.md import MolecularDynamics

from nff.utils.scatter import compute_grad
from nff.data.graphs import *
import nff.utils.constants as const


def mol_state(r, xyz):
    mass = [const.ATOMIC_MASS[item] for item in r]
    atom = "C" * r.shape[0] # intialize Atom()
    structure = Atoms(atom, positions=xyz, cell=[20.0, 20.0, 20.0], pbc=True)
    structure.set_atomic_numbers(r)
    structure.set_masses(mass)    
    return structure


def get_energy(atoms):
    """Function to print the potential, kinetic and total energy""" 
    epot = atoms.get_potential_energy() #/ len(atoms)
    ekin = atoms.get_kinetic_energy() #/ len(atoms)
    Temperature = ekin / (1.5 * units.kB * len(atoms))

    # compute kinetic energy by hand 
    # vel = torch.Tensor(atoms.get_velocities())
    # mass = atoms.get_masses()
    # mass = torch.Tensor(mass)
    # ekin = (0.5 * (vel * 1e-10 * fs * 1e15).pow(2).sum(1) * (mass * 1.66053904e-27) * 6.241509e+18).sum()
    # ekin = ekin.item() #* ev_to_kcal

    #ekin = ekin.detach().numpy()

    print('Energy per atom: Epot = %.2fkcal/mol  Ekin = %.2fkcal/mol (T=%3.0fK)  '
         'Etot = %.2fkcal/mol' % (epot * ev_to_kcal, ekin * ev_to_kcal, Temperature, (epot + ekin) * ev_to_kcal))
    # print('Energy per atom: Epot = %.5feV  Ekin = %.5feV (T=%3.0fK)  '
    #      'Etot = %.5feV' % (epot, ekin, Temperature, (epot + ekin)))
    return epot * ev_to_kcal, ekin * ev_to_kcal, Temperature


def write_traj(filename, frames):
    '''
        Write trajectory dataframes into .xyz format for VMD visualization
        to do: include multiple atom types 
        
        example:
            path = "../../sim/topotools_ethane/ethane-nvt_unwrap.xyz"
            traj2write = trajconv(n_mol, n_atom, box_len, path)
            write_traj(path, traj2write)
    '''    
    file = open(filename,'w')
    atom_no = frames.shape[1]
    for i, frame in enumerate(frames): 
        file.write( str(atom_no) + '\n')
        file.write('Atoms. Timestep: '+ str(i)+'\n')
        for atom in frame:
            if atom.shape[0] == 4:
                try:
                    file.write(str(int(atom[0])) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
                except:
                    file.write(str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write("1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()

