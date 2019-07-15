import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

import nff.utils.constants as const
from nff.md.neuralmd import NeuralMD


DEFAULT_TEMPERATURE = 450
DEFAULT_TIMESTEP = 450
DEFAULT_STEPS = 450

def NVE(species,
        xyz,
        r,
        model,
        device, 
        T=DEFAULT_TEMPERATURE,
        dt=DEFAULT_TIMESTEP,
        steps=DEFAULT_STEPS,
        output_path="./log",
        save_frequency=20,
        bond_adj=None,
        bond_len=None,
        return_pe=False):

    """function to run NVE
    
    Args:
        species (str): SMILES for the species 
        xyz (np.array): array with shape (-1, N_atom, 3)
        r (np.array): 1d np.array that consists of integers 
        model (): a Model class with pre_loaded model 
        device (int): Description
        output_path (str, optional): Description
        T (float, optional): Description
        dt (float, optional): Description
        steps (int, optional): Description
        save_frequency (int, optional): Description
    """
    # save NVE energy fluctuations, Kinetic energies and movies

    assert len(xyz.shape) == 3
    assert len(r.shape) == 2

    species_path = os.path.join(output_path, species)
    if not os.path.exists(species_path):
        os.makedirs(species_path)

    N_atom = xyz.shape[1]
    batch_size= xyz.shape[0]

    xyz = xyz.reshape(N_atom * batch_size, 3)
    r = r.reshape(-1)

    try:
        r = r.astype(int)
    except:
        raise ValueError("Z is not an array of integers")

    structure = mol_state(r=r,xyz=xyz)

    if bond_adj is not None and bond_len is not None:
        structure.set_calculator(NeuralMD(model=model, device=device, N_atom=N_atom, bond_adj=bond_adj, bond_len=bond_len))
    else:
        structure.set_calculator(NeuralMD(model=model, device=device, N_atom=N_atom))

    # Here set PBC box dimension 


    # Set the momenta corresponding to T= 0.0 K
    MaxwellBoltzmannDistribution(structure, T * units.kB)
    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(structure, dt * units.fs)
    # Now run the dynamics
    traj = []
    force_traj = []
    thermo = []
    
    n_epoch = int(steps / save_frequency)

    for i in range(n_epoch):
        dyn.run(save_frequency)
        traj.append(structure.get_positions())
        force_traj.append(dyn.atoms.get_forces())
        print("step", i * save_frequency)
        if batch_size == 1:
            epot, ekin, Temp = get_energy(structure)
            thermo.append([epot, ekin, ekin+epot, Temp])
        else:
            print("Parallelized sampling, no thermo outputs")

    traj = np.array(traj).reshape(-1, N_atom, 3)

    # save thermo data 
    thermo = np.array(thermo)
    np.savetxt(species_path + "_thermo.dat", thermo, delimiter=",")

    # write movies 
    traj = np.array(traj)
    traj = traj - traj.mean(1).reshape(-1,1,3)
    Z = np.array([r] * n_epoch).reshape(-1, N_atom, 1)
    traj_write = np.dstack(( Z, traj))

    if return_pe:
        return traj_write, np.stack(thermo[:, 0])
    else:
        return traj_write
