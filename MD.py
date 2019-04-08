from torch.autograd import Variable
from .scatter import compute_grad
from .graphs import *
import torch
import numpy as np
import os 

import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import Atoms

mass_dict = {6: 12.01, 8: 15.999, 1: 1.008, 3: 6.941, 7: 14.0067}
ev_to_kcal = 23.06035


def mol_state(r, xyz):
    mass = [mass_dict[item] for item in r]
    atom = "C" * r.shape[0] # intialize Atom()
    structure = Atoms(atom, positions=xyz, cell=[100.0, 100.0, 100.0], pbc=True)
    structure.set_atomic_numbers(r)
    structure.set_masses(mass)    
    return structure

def get_energy(atoms):
    """Function to print the potential, kinetic and total energy""" 
    epot = atoms.get_potential_energy() #/ len(atoms)
    ekin = atoms.get_kinetic_energy() #/ len(atoms)
    Temperature = ekin / (1.5 * units.kB * len(atoms))
    print('Energy per atom: Epot = %.3fkcal/mol  Ekin = %.3fkcal/mol (T=%3.0fK)  '
         'Etot = %.3fkcal/mol' % (epot * ev_to_kcal, ekin * ev_to_kcal, Temperature, (epot + ekin) * ev_to_kcal))
    return epot * ev_to_kcal, ekin* ev_to_kcal, Temperature

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
                file.write(str(int(atom[0])) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write("1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()

class NeuralMD(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device, N_atom, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.N_atom = N_atom

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        
        Calculator.calculate(self, atoms, properties, system_changes)

        # number of atoms 
        #n_atom = atoms.get_atomic_numbers().shape[0]
        N_atom = self.N_atom
        # run model 
        node = atoms.get_atomic_numbers()#.reshape(1, -1, 1)
        xyz = atoms.get_positions()#.reshape(-1, N_atom, 3)

        # to compute the kinetic energies to this...
        # (0.5 * (vel * 1e-10 * 0.09823 * 1e15).pow(2).sum(1) * (mass * 1.66054e-27) * 6.242e+18).sum()

        # rebtach based on the number of atoms 
        node = Variable(torch.LongTensor(node).reshape(-1, N_atom)).cuda(self.device)
        xyz = Variable(torch.Tensor(xyz).reshape(-1, N_atom, 3)).cuda(self.device)
        xyz.requires_grad = True

        # predict energy and force
        U = self.model(r=node, xyz=xyz)
        f_pred = -compute_grad(inputs=xyz, output=U)

        # change energy and forces back 
        U = U.reshape(-1)
        f_pred = f_pred.reshape(-1, 3)
        
        # change energy and force to numpy array 
        energy = U.detach().cpu().numpy() * (1/ev_to_kcal)
        forces = f_pred.detach().cpu().numpy() * (1/ev_to_kcal)

        #print(energy.shape)
        #print(forces.shape)
        
        self.results = {
            'energy': energy.reshape(-1),
            'forces': forces#.reshape((len(atoms), 3))
        }


def NVE(species, xyz, r, model, device, 
            dir_loc="./log", T=450.0, dt=0.1,
             steps=1000, save_frequency=20):
    """function to run NVE
    
    Args:
        species (str): smiles for the species 
        xyz (np.array): np.array that has shape (-1, N_atom, 3)
        r (np.array): 1d np.array that consists of integers 
        model (): a Model class with pre_loaded model 
        device (int): Description
        dir_loc (str, optional): Description
        T (float, optional): Description
        dt (float, optional): Description
        steps (int, optional): Description
        save_frequency (int, optional): Description
    """
    # save NVE energy fluctuations, Kinetic energies and movies

    assert len(xyz.shape) == 3
    assert len(r.shape) == 2

    if not os.path.exists(dir_loc+ "/" + species):
        os.makedirs(dir_loc + "/" + species)

    ev_to_kcal = 23.06035

    N_atom = xyz.shape[1]
    batch_size= xyz.shape[0]

    xyz = xyz.reshape(N_atom * batch_size, 3)
    r = r.reshape(-1)

    try:
        r = r.astype(int)
    except:
        raise ValueError("Z is not an array of integers")

    structure = mol_state(r=r,xyz=xyz)
    structure.set_calculator(NeuralMD(model=model, device=device, N_atom=N_atom))

    # Set the momenta corresponding to T= 0.0 K
    MaxwellBoltzmannDistribution(structure, T * units.kB)
    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(structure, dt * units.fs)
    # Now run the dynamics
    traj = []
    force_traj = []
    thermo = []
    
    n_epoch = int(steps/save_frequency)

    for i in range(n_epoch):
        dyn.run(save_frequency)
        traj.append(structure.get_positions()) # append atomic positions 
        force_traj.append(dyn.atoms.get_forces()) # append atomic forces 
        print("step", i * save_frequency)
        if batch_size == 1:
            epot, ekin, Temp = get_energy(structure)
            thermo.append([epot * ev_to_kcal, ekin * ev_to_kcal, ekin+epot, Temp])
        else:
            print("Parallelized sampling, no thermo outputs")

    traj = np.array(traj).reshape(-1, N_atom, 3)

    # save thermo data 
    thermo = np.array(thermo)
    np.savetxt(dir_loc + "/" + species + "_thermo.dat", thermo, delimiter=",")

    # write movies 
    traj = np.array(traj)
    traj = traj - traj.mean(1).reshape(-1,1,3)
    Z = np.array([r] * n_epoch).reshape(-1, N_atom, 1)
    traj_write = np.dstack(( Z, traj))

    # write forces into xyz 
    #force_traj = np.array(force_traj) * ev_to_kcal
    #Z = np.array([r] * len(force_traj)).reshape(len(force_traj), r.shape[0], 1)
    #force_write = np.dstack(( Z, force_traj))
    #write_traj(filename=dir_loc + "/" + species + "/force.xyz", frames=force_write)

    return traj_write