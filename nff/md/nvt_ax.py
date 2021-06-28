import copy
import os
import numpy as np
import math

from ase.md.md import MolecularDynamics
from ase.optimize.optimize import Dynamics
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)


class NoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 maxwell_temp=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 nbr_update_period=20,
                 append_trajectory=True,
                 **kwargs):

        if os.path.isfile(trajectory):
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
        self.ttime = ttime  # defined as a fraction of self.dt
        # Q is chosen to be 6 N kT
        self.Natom = len(atoms)

        # no rotation or translation, so target kinetic energy is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T

        self.Q = (3.0 * self.Natom - 6) * self.T * (self.ttime * self.dt)**2
        self.zeta = 0.0
        self.num_steps = max_steps
        self.n_steps = 0
        self.nbr_update_period = nbr_update_period
        self.max_steps = 0

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

        # get current acceleration and velocity:
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        vel = self.atoms.get_velocities()

        # make full step in position
        x = self.atoms.get_positions() + vel * self.dt + \
            (accel - self.zeta * vel) * (0.5 * self.dt ** 2)
        self.atoms.set_positions(x)

        # record current velocities
        KE_0 = self.atoms.get_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        # make a half step in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (self.atoms.get_kinetic_energy() - self.targeEkin)

        # make another half step in velocity
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * self.zeta)
        self.atoms.set_velocities(vel)

        return f

    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        # self.max_steps = 0

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()

class NoseHooverChain(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 num_chains,
                 maxwell_temp,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 **kwargs):

        MolecularDynamics.__init__(self,
                                   atoms,
                                   timestep * units.fs,
                                   trajectory,
                                   logfile,
                                   loginterval)

        # Initialize simulation parameters

        # convert units
        self.dt = timestep * units.fs
        self.T = temperature * units.kB

        self.N_dof = 3 * len(atoms) - 6

        # in units of fs:
        self.ttime = ttime
        self.Q = 2 * np.array([self.N_dof * self.T * (self.ttime * self.dt)**2,
                               *[self.T * (self.ttime * self.dt)**2]*(num_chains-1)])

        # no rotation or translation, so target kinetic energy is 3/2 N kT - 6
        self.targeEkin = 1/2 * self.N_dof * self.T

        # self.zeta = np.array([0.0]*num_chains)
        self.p_zeta = np.array([0.0]*num_chains)
        self.num_steps = max_steps
        self.n_steps = 0
        self.max_steps = 0

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is not None:
            # convert units
            maxwell_temp = maxwell_temp * units.kB
        else:
            maxwell_temp = 2 * self.T
        MaxwellBoltzmannDistribution(self.atoms, maxwell_temp)

    def get_zeta_accel(self):

        p0_dot = 2 * (self.atoms.get_kinetic_energy() - self.targeEkin) - \
            self.p_zeta[0]*self.p_zeta[1] / self.Q[1]
        p_middle_dot = self.p_zeta[:-2]**2 / self.Q[:-2] - \
            self.T - self.p_zeta[1:-1] * self.p_zeta[2:]/self.Q[2:]
        p_last_dot = self.p_zeta[-2]**2 / self.Q[-2] - self.T
        p_dot = np.array([p0_dot, *p_middle_dot, p_last_dot])

        return p_dot / self.Q

    def half_step_v_zeta(self):

        v = self.p_zeta / self.Q
        accel = self.get_zeta_accel()
        v_half = v + 1/2 * accel * self.dt
        return v_half

    def half_step_v_system(self):

        v = self.atoms.get_velocities()
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        accel -= v * self.p_zeta[0] / self.Q[0]
        v_half = v + 1/2 * accel * self.dt
        return v_half

    def full_step_positions(self):

        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        new_positions = self.atoms.get_positions() + self.atoms.get_velocities() * self.dt + \
            (accel - self.p_zeta[0] / self.Q[0])*(self.dt)**2
        return new_positions

    def step(self):

        new_positions = self.full_step_positions()
        self.atoms.set_positions(new_positions)

        v_half_system = self.half_step_v_system()
        v_half_zeta = self.half_step_v_zeta()

        self.atoms.set_velocities(v_half_system)
        self.p_zeta = v_half_zeta * self.Q

        v_full_zeta = self.half_step_v_zeta()
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        v_full_system = (v_half_system + 1/2 * accel * self.dt) / \
            (1 + 0.5 * self.dt * v_full_zeta[0])

        self.atoms.set_velocities(v_full_system)
        self.p_zeta = v_full_zeta * self.Q

    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        # self.max_steps = 0

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()
