import os 
import numpy as np
import torch
from torch.autograd import Variable

from ase import units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

import nff.utils.constants as const
from nff.md.utils import NeuralMDLogger, write_traj
from nff.io.ase import NeuralFF

DEFAULTNVEPARAMS = {'T_init': 300.0, 
            'time_step': 0.5, 
            'thermostat': VelocityVerlet, # or Langevin or NPT or NVT or Thermodynamic Integration
            'nbr_list_update_freq': 50,
            'steps': 100}

DEFAULTLOGPARAMS = {'save_frequency': 20,
                    'thermo_filename': './thermo.log', 
                    'traj_filename': './atom.traj'}


class Dynamics:
    
    def __init__(self, 
                atomsbatch,
                models,
                mdparam=DEFAULTNVEPARAMS,
                logparam=DEFAULTLOGPARAMS,
                ):
    
        # initialize the atoms batch system 
        self.calculator = NeuralFF(model=model, props=atomsbatch.get_batch())
        self.atomsbatch = atomsbatch
        self.mdparam = mdparam
        self.logparam = logparam
   
        # add calculators 
        self.atomsbatch.set_calculator(self.calculator)
        
        # integration method 
        integrator = self.mdparam['thermostat']
        
        # todo: structure optimization before starting
        
        # intialize system momentum 
        MaxwellBoltzmannDistribution(self.atomsbatch, self.mdparam['T_init'] * units.kB)
        
        # set thermostats 
        integrator = self.mdparam['thermostat']
        
        self.integrator = integrator(self.atomsbatch, 
                                    self.mdparam['time_step'] * units.fs)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.logparam['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=logparam['save_frequency'])
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, 
                                        self.atomsbatch, 
                                        self.logparam['thermo_filename'], 
                                        mode="a"), interval=logparam['save_frequency'])
    
    def run(self):
        
        self.integrator.run(self.mdparam['steps'])
        
    
    def save_as_xyz(self):
        
        '''
        TODO: save time information 
        TODO: subclass TrajectoryReader/TrajectoryReader to digest AtomsBatch instead of Atoms?
        TODO: other system variables in .xyz formats 
        '''
        traj = Trajectory(self.logparam['traj_filename'], mode='r')
        
        xyz = []
        
        for snapshot in traj:
            frames = np.concatenate([
            snapshot.get_atomic_numbers().reshape(-1, 1),
            snapshot.get_positions().reshape(-1, 3)
            ], axis=1)
            
            xyz.append(frames)
            
        write_traj('./traj.xyz', np.array(xyz))