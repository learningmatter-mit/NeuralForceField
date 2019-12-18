import os 
import numpy as np

from ase.md.md import MolecularDynamics


class NoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 trajectory=None,
                 logfile=None,
                 loginterval=1):

        MolecularDynamics.__init__(self,
                                   atoms,
                                   timestep,
                                   trajectory,
                                   logfile,
                                   loginterval)
        
        # Initialize simulation parameters 

        # Q is chosen to be 6 N kT 
        self.dt = timestep 
        self.Natom = atoms.get_number_of_atoms()
        self.T = temperature
        self.targeEkin = 0.5 * (3.0 * self.Natom ) * self.T
        self.ttime = ttime #* units.fs
        self.Q =  3.0 * self.Natom * self.T * (self.ttime * self.dt) ** 2
        self.zeta = 0.0
    
    def step(self, f=None):
        
        # get current acceleration and velocity: 
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        vel = self.atoms.get_velocities()

        # make full step in position 
        x = self.atoms.get_positions() + vel * self.dt + (accel - self.zeta * vel) * (0.5 * self.dt ** 2)
        self.atoms.set_positions(x)

        #record current velocities 
        KE_0 = self.atoms.get_kinetic_energy()

        # make half a step in velocity 
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        # make a half step in self.zeta 
        self.zeta = self.zeta + 0.5 * self.dt * (1/self.Q) * (KE_0 -  self.targeEkin)

        # make another halfstep in self.zeta 
        self.zeta = self.zeta + 0.5 * self.dt * (1/self.Q)  * (self.atoms.get_kinetic_energy() - self.targeEkin)

        # make another half step in velocity
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel )/(1 + 0.5 * self.dt * self.zeta)
        self.atoms.set_velocities(vel)
        
        return f
