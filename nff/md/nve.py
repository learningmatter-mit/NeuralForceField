import numpy as np
import os

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary,
                                         ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory

from nff.md.utils import NeuralMDLogger, write_traj

DEFAULTNVEPARAMS = {
    'T_init': 120.0,
    # thermostat can be NoseHoover, Langevin, NPT, NVT,
    # Thermodynamic Integration...
    'thermostat': VelocityVerlet,
    'thermostat_params': {'timestep': 0.5 * units.fs},
    'nbr_list_update_freq': 20,
    'steps': 3000,
    'save_frequency': 10,
    'thermo_filename': './thermo.log',
    'traj_filename': './atoms.traj',
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
        MaxwellBoltzmannDistribution(self.atomsbatch,
                                     temperature_K = self.mdparam['T_init'])
        Stationary(self.atomsbatch)  # zero linear momentum
        ZeroRotation(self.atomsbatch)
        
        # set thermostats
        integrator = self.mdparam['thermostat']
        if integrator == VelocityVerlet:
            dt = self.mdparam['thermostat_params']['timestep']
            self.integrator = integrator(self.atomsbatch,
                                         timestep=dt)
        else:
            self.integrator = integrator(self.atomsbatch,
                                         **self.mdparam['thermostat_params'],
                                         **self.mdparam)
        
        self.steps = int(self.mdparam['steps'])
        self.check_restart()
        
        if self.steps == int(self.mdparam['steps']):
            # attach trajectory dump
            self.traj = Trajectory(
                self.mdparam['traj_filename'], 'w', self.atomsbatch)
            self.integrator.attach(
                self.traj.write, interval=self.mdparam['save_frequency'])

            # attach log file
            requires_stress = 'stress' in self.atomsbatch.calc.properties
            self.integrator.attach(NeuralMDLogger(self.integrator,
                                                self.atomsbatch,
                                                self.mdparam['thermo_filename'],
                                                stress=requires_stress,
                                                mode='a'),
                                interval=self.mdparam['save_frequency'])
        
    def check_restart(self):

        if os.path.exists(self.mdparam['traj_filename']):
            new_atoms = Trajectory(self.mdparam['traj_filename'])[-1]

            # calculate number of steps remaining
            self.steps = ( int(self.mdparam['steps']) 
                - ( int(self.mdparam['save_frequency']) * 
                len(Trajectory(self.mdparam['traj_filename'])) ) )

            self.atomsbatch.set_cell(new_atoms.get_cell())
            self.atomsbatch.set_positions(new_atoms.get_positions())
            self.atomsbatch.set_velocities(new_atoms.get_velocities())

            # attach trajectory dump
            self.traj = Trajectory(
                self.mdparam['traj_filename'], 'a', self.atomsbatch)
            self.integrator.attach(
                self.traj.write, interval=self.mdparam['save_frequency'])

            # attach log file
            requires_stress = 'stress' in self.atomsbatch.calc.properties
            self.integrator.attach(NeuralMDLogger(self.integrator,
                                                self.atomsbatch,
                                                self.mdparam['thermo_filename'],
                                                stress=requires_stress,
                                                mode='a'),
                                interval=self.mdparam['save_frequency'])     

            return self.steps

        else:

            return


    def setup_restart(self, restart_param):
        """If you want to restart a simulations with predfined mdparams but 
        longer you need to provide a dictionary like the following:

         note that the thermo_filename and traj_name should be different 

         restart_param = {'atoms_path': md_log_dir + '/atom.traj', 
                          'thermo_filename':  md_log_dir + '/thermo_restart.log',
                          'traj_filename': md_log_dir + '/atom_restart.traj',
                          'steps': 100
                          }

        Args:
            restart_param (dict): dictionary to contains restart paramsters and file paths
        """

        if restart_param['thermo_filename'] == self.mdparam['thermo_filename']:
            raise ValueError("{} is also used, \
                please change a differnt thermo file name".format(restart_param['thermo_filename']))

        if restart_param['traj_filename'] == self.mdparam['traj_filename']:
            raise ValueError("{} is also used, \
                please change a differnt traj file name".format(restart_param['traj_filename']))

        self.restart_param = restart_param
        new_atoms = Trajectory(restart_param['atoms_path'])[-1]

        self.atomsbatch.set_positions(new_atoms.get_positions())
        self.atomsbatch.set_velocities(new_atoms.get_velocities())

        # set thermostats
        integrator = self.mdparam['thermostat']
        self.integrator = integrator(
            self.atomsbatch, **self.mdparam['thermostat_params'])

        # attach trajectory dump
        self.traj = Trajectory(
            self.restart_param['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(
            self.traj.write, interval=self.mdparam['save_frequency'])

        # attach log file
        requires_stress = 'stress' in self.atomsbatch.calc.properties
        self.integrator.attach(NeuralMDLogger(
            self.integrator,
            self.atomsbatch,
            self.restart_param['thermo_filename'],
            stress=requires_stress,
            mode='a'),
            interval=self.mdparam['save_frequency'])

        self.mdparam['steps'] = restart_param['steps']

    def run(self):

        epochs = int(self.steps //
                     self.mdparam['nbr_list_update_freq'])
        # In case it had neighbors that didn't include the cutoff skin,
        # for example, it's good to update the neighbor list here
        self.atomsbatch.update_nbr_list()

        for step in range(epochs):
            self.integrator.run(self.mdparam['nbr_list_update_freq'])

            # # unwrap coordinates if mol_idx is defined
            # if self.atomsbatch.props.get("mol_idx", None) :
            #     self.atomsbatch.set_positions(self.atoms.get_positions(wrap=True))
            #     self.atomsbatch.set_positions(reconstruct_atoms(atoms, self.atomsbatch.props['mol_idx']))

            self.atomsbatch.update_nbr_list()

        self.traj.close()

    def save_as_xyz(self, filename='./traj.xyz'):

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
