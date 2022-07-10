import os
import numpy as np
import copy
import math

from tqdm import tqdm
from ase.optimize.optimize import Dynamics
from ase.md.md import MolecularDynamics
from ase.md.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from nff.io.ase import AtomsBatch


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
        self.ttime = ttime  # defined as a fraction of self.dt
        # Q is chosen to be 6 N kT
        self.Natom = len(atoms)

        # no rotation or translation, so target kinetic energy
        # is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T

        self.Q = (3.0 * self.Natom - 6) * self.T * (self.ttime * self.dt)**2
        self.zeta = 0.0
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
        self.remove_constrained_vel(atoms)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)

    def remove_constrained_vel(self, atoms):
        """
        Set the initial velocity to zero for any constrained or fixed atoms
        """

        constraints = atoms.constraints
        fixed_idx = []
        for constraint in constraints:
            has_keys = False
            keys = ['idx', 'indices', 'index']
            for key in keys:
                if hasattr(constraint, key):
                    val = np.array(getattr(constraint, key)
                                   ).reshape(-1).tolist()
                    fixed_idx += val
                    has_keys = True
            if not has_keys:
                print(("WARNING: velocity not set to zero for any atoms in constraint "
                       "%s; do not know how to find its fixed indices." % constraint))

        if not fixed_idx:
            return

        fixed_idx = np.array(list(set(fixed_idx)))
        vel = self.atoms.get_velocities()
        vel[fixed_idx] = 0
        self.atoms.set_velocities(vel)

    def step(self):

        # get current acceleration and velocity:

        accel = (self.atoms.get_forces() /
                 self.atoms.get_masses().reshape(-1, 1))

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
            (1 / self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1 / self.Q) * (self.atoms.get_kinetic_energy() - self.targeEkin)

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
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()


class NoseHooverChain(NoseHoover):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 num_chains,
                 maxwell_temp=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 nbr_update_period=20,
                 append_trajectory=True,
                 **kwargs):

        NoseHoover.__init__(self,
                            atoms=atoms,
                            timestep=timestep,
                            temperature=temperature,
                            ttime=ttime,
                            maxwell_temp=maxwell_temp,
                            trajectory=trajectory,
                            logfile=logfile,
                            loginterval=loginterval,
                            max_steps=max_steps,
                            nbr_update_period=nbr_update_period,
                            append_trajectory=append_trajectory,
                            **kwargs)

        self.N_dof = 3.0 * self.Natom - 6
        q_0 = self.N_dof * self.T * (self.ttime * self.dt) ** 2
        q_n = self.T * (self.ttime * self.dt) ** 2

        self.Q = 2 * np.array([q_0, *([q_n] * (num_chains-1))])
        self.p_zeta = np.array([0.0]*num_chains)

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
        accel = (self.atoms.get_forces()
                 / self.atoms.get_masses().reshape(-1, 1))
        accel -= v * self.p_zeta[0] / self.Q[0]
        v_half = v + 1/2 * accel * self.dt
        return v_half

    def full_step_positions(self):

        accel = (self.atoms.get_forces()
                 / self.atoms.get_masses().reshape(-1, 1))
        new_positions = (self.atoms.get_positions() +
                         self.atoms.get_velocities() * self.dt
                         (accel - self.p_zeta[0] / self.Q[0]) * (self.dt)**2)
        return new_positions

    def step(self):

        new_positions = self.full_step_positions()
        self.atoms.set_positions(new_positions)

        v_half_system = self.half_step_v_system()
        v_half_zeta = self.half_step_v_zeta()

        self.atoms.set_velocities(v_half_system)
        self.p_zeta = v_half_zeta * self.Q

        v_full_zeta = self.half_step_v_zeta()
        accel = (self.atoms.get_forces()
                 / self.atoms.get_masses().reshape(-1, 1))
        v_full_system = (v_half_system + 1/2 * accel * self.dt) / \
            (1 + 0.5 * self.dt * v_full_zeta[0])

        self.atoms.set_velocities(v_full_system)
        self.p_zeta = v_full_zeta * self.Q
        
        
class NoseHooverChainsBiased(NoseHooverChain):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 num_chains,
                 maxwell_temp=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 nbr_update_period=20,
                 append_trajectory=True,
                 **kwargs):

        NoseHooverChain.__init__(self,
                            atoms=atoms,
                            timestep=timestep,
                            temperature=temperature,
                            ttime=ttime,
                            num_chains=num_chains,
                            maxwell_temp=maxwell_temp,
                            trajectory=trajectory,
                            logfile=logfile,
                            loginterval=loginterval,
                            max_steps=max_steps,
                            nbr_update_period=nbr_update_period,
                            append_trajectory=append_trajectory,
                            **kwargs)


    def update_bias(self):
        # update the bias function if necessary, e.g., add aconfiguration to MetaD
        self.atoms.calc.update(self)

    def irun(self):
        # run the algorithm max_steps reached
        while self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1
            self.update_bias()
            
            # log the step
            self.log()
            self.call_observers()

        
    def run(self, steps=None):
        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update

        self.atoms.update_nbr_list()
        
        # compute initial structure and log the first step
        if self.nsteps == 0:
            self.update_bias()
            self.atoms.get_forces()
            self.log()
            self.call_observers()
            
        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            self.irun()
            self.atoms.update_nbr_list()

            
class Langevin(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep: float,
                 temperature: float,
                 friction_per_ps: float = 1.0,
                 maxwell_temp: float = None,
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
        self.T  = temperature 
        
        self.friction  = friction_per_ps * 1.0e-3 / units.fs
        self.rand_push = (np.sqrt(self.T * self.friction * self.dt * units.kB / 2.0e0) / 
                            np.sqrt(self.atoms.get_masses().reshape(-1,1)))
        self.prefac1   = 2.0 / (2.0 + self.friction * self.dt)
        self.prefac2   = (2.0e0 - self.friction * self.dt) / (2.0e0 + self.friction * self.dt)

        self.num_steps = max_steps
        self.n_steps   = 0
        self.max_steps = 0

        self.nbr_update_period = nbr_update_period

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is not None:
            maxwell_temp = maxwell_temp
        else:
            maxwell_temp = self.T

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=maxwell_temp)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.remove_constrained_vel(atoms)

    def remove_constrained_vel(self, atoms):
        """
        Set the initial velocity to zero for any constrained or fixed atoms
        """

        constraints = atoms.constraints
        fixed_idx = []
        for constraint in constraints:
            has_keys = False
            keys = ['idx', 'indices', 'index']
            for key in keys:
                if hasattr(constraint, key):
                    val = np.array(getattr(constraint, key)
                                   ).reshape(-1).tolist()
                    fixed_idx += val
                    has_keys = True
            if not has_keys:
                print(("WARNING: velocity not set to zero for any atoms in constraint "
                       "%s; do not know how to find its fixed indices." % constraint))

        if not fixed_idx:
            return

        fixed_idx = np.array(list(set(fixed_idx)))
        vel = self.atoms.get_velocities()
        vel[fixed_idx] = 0
        self.atoms.set_velocities(vel)

    def step(self):
        
        vel    = self.atoms.get_velocities()
        masses = self.atoms.get_masses().reshape(-1,1)
        
        self.rand_gauss = np.random.randn(self.atoms.get_positions().shape[0],
                                          self.atoms.get_positions().shape[1]) 

        vel +=  self.rand_push * self.rand_gauss
        vel += 0.5e0 * self.dt * self.atoms.get_forces() / masses
        
        self.atoms.set_velocities(vel)
        self.remove_constrained_vel(self.atoms)
        
        vel  = self.atoms.get_velocities()
        x    = self.atoms.get_positions() + self.prefac1 * self.dt * vel

        # update positions
        self.atoms.set_positions(x)

        vel *= self.prefac2
        vel += self.rand_push * self.rand_gauss
        vel += 0.5e0 * self.dt * self.atoms.get_forces() / masses

        self.atoms.set_velocities(vel)
        self.remove_constrained_vel(self.atoms)

    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()
            Stationary(self.atoms)
            ZeroRotation(self.atoms)
  


class NoseHooverMetadynamics(NoseHoover):
    def __init__(self,
                 atomsbatch,
                 timestep,
                 temperature,
                 ttime,
                 geom_add_time,
                 max_steps=None,
                 trajectory="mtd.trj",
                 logfile="mtd.log",
                 loginterval=1,
                 **kwargs):

        NoseHoover.__init__(self,
                            atoms=atomsbatch,
                            timestep=timestep,
                            temperature=temperature,
                            ttime=ttime,
                            trajectory=trajectory,
                            logfile=logfile,
                            loginterval=loginterval,
                            max_steps=max_steps,
                            **kwargs)

        self.geom_add_time = geom_add_time * units.fs
        self.max_steps = 0

    def increase_h_mass(self):
        nums = self.atoms.get_atomic_numbers()
        masses = self.atoms.get_masses()
        masses[nums == 1] = 4.0
        self.atoms.set_masses(masses)

    def decrease_h_mass(self):
        nums = self.atoms.get_atomic_numbers()
        masses = self.atoms.get_masses()
        masses[nums == 1] = 1.008
        self.atoms.set_masses(masses)

    def append_atoms(self):
        # Create new atoms so we don't copy the model on the calculator of the atoms
        # No need to set other attributes equal to those of `self.atoms`, since (1)
        # it's unnecessary, and (2) it leads to `new_atoms` being updated every time
        # `self.atoms` is (not sure why)
        new_atoms = AtomsBatch(numbers=self.atoms.get_atomic_numbers(),
                               positions=self.atoms.get_positions())

        self.atoms.calc.append_atoms(new_atoms)

    def run(self, steps=None):
        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        # self.max_steps = 0

        # number of steps until we add a new geom

        steps_between_add = int(self.geom_add_time / self.dt)
        steps_until_add = copy.deepcopy(steps_between_add)

        self.atoms.update_nbr_list()
        self.append_atoms()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            # set hydrogen mass to 2 AMU (deuterium, following Grimme's mTD approach)
            self.increase_h_mass()

            Dynamics.run(self)

            # reset the masses
            self.decrease_h_mass()

            self.atoms.update_nbr_list()

            if self.nsteps >= steps_until_add:
                self.append_atoms()
                steps_until_add += steps_between_add


class BatchNoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 T_init=None,
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

        # Q is chosen to be 6 N kT
        self.dt = timestep * units.fs
        self.Natom = len(atoms)
        self.T = temperature * units.kB

        # no rotation or translation, so target kinetic energy
        # is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T
        self.ttime = ttime  # * units.fs

        self.zeta = 0.0
        self.num_steps = max_steps
        self.n_steps = 0
        self.nbr_update_period = nbr_update_period
        self.max_steps = 0

        batch = atoms.get_batch()

        # Check for number of virtual variables
        if batch.get('num_atoms', None) is not None:
            self.Natom = batch.get('num_atoms', None).numpy()
            self.n_sys = self.Natom.shape[0]
            self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T
        else:
            self.n_sys = 1

        self.Q = np.array(
            (3.0 * self.Natom - 6) * self.T * (self.ttime * self.dt)**2)
        self.zeta = np.array([0.0] * self.n_sys)

        if T_init is None:
            T_init = self.T / units.kB

        # intialize system momentum
        momenta = []
        # split AtomsBatch into separate Atoms objects

        for atoms in self.atoms.get_list_atoms():
            # set MaxwellBoltzmannDistribution for each Atoms objects separately

            MaxwellBoltzmannDistribution(atoms,
                                         temperature_K=T_init)
            Stationary(atoms)  # zero linear momentum
            ZeroRotation(atoms)

            # set momenta for the individual Atoms objects within the AtomsBatch
            momenta.append(atoms.get_momenta())

        momenta = np.concatenate(momenta)
        self.atoms.set_momenta(momenta)

    def step(self):

        # get current acceleration and velocity
        accel = (self.atoms.get_forces() /
                 self.atoms.get_masses().reshape(-1, 1))
        vel = self.atoms.get_velocities()
        zeta_vel = np.repeat(self.zeta, 3 * self.Natom).reshape(-1, 3) * vel
        visc = accel - zeta_vel

        # make full step in position
        x = self.atoms.get_positions() + vel * self.dt + \
            visc * (0.5 * self.dt ** 2)
        self.atoms.set_positions(x)

        # record current velocities
        KE_0 = self.atoms.get_batch_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * visc
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        # make a half step in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1 / self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = (self.zeta + 0.5 * self.dt * (1 / self.Q) *
                     (self.atoms.get_batch_kinetic_energy() -
                      self.targeEkin))

        # make another half step in velocity
        scal = (1 + 0.5 * self.dt * self.zeta)

        vel = self.atoms.get_velocities()
        vel = ((vel + 0.5 * self.dt * accel) /
               np.repeat(scal, 3 * self.Natom).reshape(-1, 3))

        self.atoms.set_velocities(vel.reshape(-1, 3))

        return f

    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update

        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()


class BatchMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 **kwargs):

        kwargs['header'] = False
        MDLogger.__init__(self,
                          *args,
                          **kwargs)

        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""

        num_atoms = self.atoms.num_atoms

        for i in range(len(num_atoms)):
            if i != 0:
                self.hdr += " "
            self.hdr += "    %12s    %6s" % ("Epot{num} [eV]", "T{num} [K]")
            self.hdr = self.hdr.format(num=(i + 1))

        digits = 4
        for _ in range(len(num_atoms)):
            self.fmt += 1 * ("%%12.%df " % (digits,)) + " %9.1f"
            self.fmt += "  "

        self.fmt += "\n"

        self.logfile.write(self.hdr + "\n")

    def __call__(self):
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()

        ekin = self.atoms.get_batch_kinetic_energy()
        epot = self.atoms.get_potential_energy()
        temp = self.atoms.get_batch_T()

        for i, this_ek in enumerate(ekin):
            this_epot = epot[i]
            this_temp = float(temp[i])
            dat += (this_epot, this_temp)

        self.logfile.write(self.fmt % dat)
        self.logfile.flush()


class BatchNoseHooverMetadynamics(BatchNoseHoover,
                                  NoseHooverMetadynamics):
    def __init__(self,
                 atomsbatch,
                 timestep,
                 temperature,
                 ttime,
                 geom_add_time,
                 max_steps=None,
                 trajectory="mtd.trj",
                 logfile="mtd.log",
                 loginterval=1,
                 **kwargs):

        # follow Nose-Hoover here and default to 2 * T

        if kwargs.get("maxwell_temp") is not None:
            kwargs["T_init"] = kwargs["maxwell_temp"]
        else:
            kwargs["T_init"] = 2 * temperature

        BatchNoseHoover.__init__(self,
                                 atoms=atomsbatch,
                                 timestep=timestep,
                                 temperature=temperature,
                                 ttime=ttime,
                                 trajectory=trajectory,
                                 logfile=None,
                                 loginterval=loginterval,
                                 max_steps=max_steps,
                                 **kwargs)

        self.geom_add_time = geom_add_time * units.fs
        self.max_steps = 0

        if logfile:
            logger = BatchMDLogger(dyn=self,
                                   atoms=atomsbatch,
                                   logfile=logfile)
            self.attach(logger, loginterval)

    def run(self, steps=None):
        NoseHooverMetadynamics.run(self, steps)
