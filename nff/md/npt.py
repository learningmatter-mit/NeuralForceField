import os
import numpy as np
import copy
import math
from tqdm import tqdm
from ase.md.md import MolecularDynamics
from ase.optimize.optimize import Dynamics
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)


class NoseHoovernpt(NPT):
    def __init__(self, atoms,
                 timestep,
                 temperature=None,
                 externalstress=None,
                 ttime=None,
                 T_init=None,
                 pfactor=None,
                 mask=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 nbr_update_period=20,
                 append_trajectory=False,
                 **kwargs):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        print(externalstress)

        NPT.__init__(self,atoms=atoms,
                     timestep=timestep * units.fs,
                     ttime=ttime,
                     externalstress=externalstress,
                     pfactor=pfactor,
                     temperature_K=temperature,
                     mask=mask,
                     trajectory=trajectory,
                     logfile=logfile,
                     loginterval=loginterval,
                     append_trajectory=append_trajectory)

        # Initialize simulation parameters
        # convert units
        self.nbr_update_period = nbr_update_period
        self.max_steps=0

        self.T = temperature * units.kB

        # initial Maxwell-Boltmann temperature for atoms
        if T_init is not None:
            # convert units
            T_init = T_init * units.kB
        else:
            T_init = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, temperature*units.kB)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.initialize()


    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()


            
class NoseHooverNPT(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 pressure,
                 ttime,
                 Pdamp,
                 maxwell_temp=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 nbr_update_period=20,
                 append_trajectory=True,
                 **kwargs):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        MolecularDynamics.__init__(self,
                                   atoms=atoms,
                                   timestep=timestep * units.fs,
                                   trajectory=trajectory,
                                   logfile=logfile,
                                   loginterval=loginterval,
                                   append_trajectory=append_trajectory)

        # Initialize simulation parameters
        # convert units

        self.dt = timestep * units.fs
        self.T = temperature * units.kB
        self.P= pressure*units.GPa
        self.ttime = ttime  # defined as a fraction of self.dt
        self.pdamp= Pdamp
        # Q is chosen to be 6 N kT
        self.Natom = len(atoms)
        self.Nf =3*self.Natom - 6

        # no rotation or translation, so target kinetic energy
        # is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (self.Nf) * self.T

        self.Q = (self.Nf ) * self.T * (self.ttime * self.dt)**2
        self.W = (self.Natom-1)* self.T *(self.pdamp*self.dt)**2
        self.zeta = 0.0
        self.eta=0.0
        self.veta=0.0
        self.num_steps = max_steps
        self.n_steps = 0
        self.max_steps = 0

        self.nbr_update_period = nbr_update_period

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is not None:
            # convert units
            maxwell_temp = maxwell_temp * units.kB
        else:
            maxwell_temp = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, maxwell_temp)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
    def step(self):
        accel = (self.atoms.get_forces() /
                 self.atoms.get_masses().reshape(-1, 1))

        vel = self.atoms.get_velocities()
        Pint=-np.sum(self.atoms.get_stress(include_ideal_gas=True)[0:3])/3
        F=3*self.atoms.get_volume()*(Pint-self.P) + (6/self.Nf)*self.atoms.get_kinetic_energy()
        G= (1/self.Q)*(2*self.atoms.get_kinetic_energy()+self.W*(self.veta**2)-(self.Nf+1)*self.T)
        eta0=self.eta
        self.eta =self.eta + self.veta * self.dt + 0.5 *((F/self.W)-self.veta*self.zeta)*self.dt*self.dt
        x = np.exp(self.eta-eta0)*(self.atoms.get_positions() + vel * self.dt + \
            (accel - self.zeta * vel - (2+(3/self.Nf))*vel*self.veta) * (0.5 * self.dt ** 2))
        self.atoms.set_positions(x)
        # make half a step in velocity
        vel_half = np.exp(self.eta-eta0)*(vel + 0.5 * self.dt * (accel - self.zeta * vel- (2+(3/self.Nf))*vel*self.veta))
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)
        self.zeta = self.zeta + 0.5 * self.dt * G
        self.veta=self.veta+  0.5 * self.dt *((F/self.W)-self.veta*self.zeta)
        Vol=self.atoms.get_volume()*np.exp(3*self.eta-3*eta0)
        h=Vol**(1/3)
        self.atoms.set_cell([h,h,h])
        
        Pint=-np.sum(self.atoms.get_stress(include_ideal_gas=True)[0:3])/3
        F=3*self.atoms.get_volume()*(Pint-self.P) + (6/self.Nf)*self.atoms.get_kinetic_energy()
        G= (1/self.Q)*(2*self.atoms.get_kinetic_energy()+self.W*(self.veta**2)-(self.Nf+1)*self.T)
        self.zeta = self.zeta + 0.5 * self.dt * G
        self.veta= (self.veta + 0.5*self.dt*(F/self.W))/(1 + 0.5 * self.dt * self.zeta)
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * self.zeta + 0.5* self.dt *(2+(3/self.Nf))*self.veta)
        self.atoms.set_velocities(vel)
        #Vol=self.atoms.get_volume()*np.exp(3*self.eta-3*eta0)
        #h=Vol**(1/3)
        #self.atoms.set_cell([h,h,h])

        return f
    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()
         
