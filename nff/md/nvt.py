
import os
import numpy as np
import pdb

from ase.md.md import MolecularDynamics


class NoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 **kwargs):

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
        self.targeEkin = 0.5 * (3.0 * self.Natom) * self.T
        self.ttime = ttime  # * units.fs
        self.Q = 3.0 * self.Natom * self.T * (self.ttime * self.dt)**2
        self.zeta = 0.0

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


class NoseHooverChain(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 num_chains,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 **kwargs):

        MolecularDynamics.__init__(self,
                                   atoms,
                                   timestep,
                                   trajectory,
                                   logfile,
                                   loginterval)

        # Initialize simulation parameters

        self.dt = timestep

        self.N_dof = 3*atoms.get_number_of_atoms()
        self.T = temperature

        # in units of fs:
        self.ttime = ttime
        self.Q = 2 * np.array([self.N_dof * self.T * (self.ttime * self.dt)**2,
                           *[self.T * (self.ttime * self.dt)**2]*(num_chains-1)])
        self.targeEkin = 1/2 * self.N_dof * self.T

        # self.zeta = np.array([0.0]*num_chains)
        self.p_zeta = np.array([0.0]*num_chains)

    def get_zeta_accel(self):

        p0_dot = 2 * (self.atoms.get_kinetic_energy() - self.targeEkin)- \
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



