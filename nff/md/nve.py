import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory

import nff.utils.constants as const
from nff.md.utils import NeuralMDLogger, write_traj
from nff.io.ase import NeuralFF

DEFAULTNVEPARAMS = {
    'T_init': 120.0, 
#     'thermostat': NoseHoover,   # or Langevin or NPT or NVT or Thermodynamic Integration
#     'thermostat_params': {'timestep': 0.5 * units.fs, "temperature": 120.0 * units.kB,  "ttime": 20.0}
    'thermostat': VelocityVerlet,  
    'thermostat_params': {'timestep': 0.5 * units.fs},
    'nbr_list_update_freq': 20,
    'steps': 3000,
    'save_frequency': 10,
    'thermo_filename': './thermo.log', 
    'traj_filename': './atom.traj',
    'skip': 0
}


class Dynamics:
    
    def __init__(self, 
                atomsbatch,
                mdparam=DEFAULTNVEPARAMS,
                ):
    
        # initialize the atoms batch system 
        self.atomsbatch = atomsbatch
        self.mdparam = mdparam
   
        # todo: structure optimization before starting
        
        # intialize system momentum 
        MaxwellBoltzmannDistribution(self.atomsbatch, self.mdparam['T_init'] * units.kB)
        
        # set thermostats 
        integrator = self.mdparam['thermostat']
        
        self.integrator = integrator(self.atomsbatch, **self.mdparam['thermostat_params'])
        
        # attach trajectory dump 
        self.traj = Trajectory(self.mdparam['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=mdparam['save_frequency'])
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, 
                                        self.atomsbatch, 
                                        self.mdparam['thermo_filename'], 
                                        mode='a'), interval=mdparam['save_frequency'])
    
    def run(self):
        # 
        epochs = int(self.mdparam['steps'] // self.mdparam['nbr_list_update_freq'])
        
        for step in range(epochs):
            self.integrator.run(self.mdparam['nbr_list_update_freq'])
            self.atomsbatch.update_nbr_list()

        self.traj.close()
        
    
    def save_as_xyz(self, filename='./traj.xyz'):
        
        '''
        TODO: save time information 
        TODO: subclass TrajectoryReader/TrajectoryReader to digest AtomsBatch instead of Atoms?
        TODO: other system variables in .xyz formats 
        '''
        traj = Trajectory(self.mdparam['traj_filename'], mode='r')
        
        xyz = []
        
        skip = self.mdparam['skip']
        traj = list(traj)[skip:] if len(traj) > skip else traj

        for snapshot in traj:
            frames = np.concatenate([
                snapshot.get_atomic_numbers().reshape(-1, 1),
                snapshot.get_positions().reshape(-1, 3)
            ], axis=1)
            
            xyz.append(frames)
            
        write_traj(filename, np.array(xyz))