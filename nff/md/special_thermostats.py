import os
import numpy as np

from tqdm import tqdm
from ase.optimize.optimize import Dynamics
from ase.md.md import MolecularDynamics
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


class TempRamp(MolecularDynamics):
    def __init__(
        self,
        atoms,
        timestep,
        target_temp,
        num_steps,
        maxwell_temp=None,
        trajectory=None,
        logfile=None,
        loginterval=1,
        nbr_update_period=20,
        append_trajectory=True,
        **kwargs,
    ):
        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        MolecularDynamics.__init__(
            self,
            atoms=atoms,
            timestep=timestep * units.fs,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

        # Initialize simulation parameters
        # convert units

        self.dt = timestep * units.fs
        self.Natom = len(atoms)

        if self.atoms.pbc:
            self.activeDoF = (3 * self.Natom) - len(self.atoms.constraints)
        else:
            # no rotation or translation, so target kinetic energy
            self.activeDoF = (3 * self.Natom) - 6 - len(self.atoms.constraints)

        self.num_steps = num_steps
        self.nbr_update_period = nbr_update_period
        if self.num_steps < self.nbr_update_period:
            print("WARNING: Ramp will be performed in a single rescaling step!")
        if self.num_steps % self.nbr_update_period != 0:
            print(
                "WARNING: Number of steps is adjusted to "
                f"{self.num_steps + self.nbr_update_period - (self.num_steps % self.nbr_update_period)}!"
            )

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is not None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=maxwell_temp)
            self.start_temp = maxwell_temp
        else:
            self.start_temp = (2 * self.atoms.get_kinetic_energy()) / (units.kB * self.activeDoF)

        self.num_epochs = int(np.ceil(self.num_steps / self.nbr_update_period))
        self.ramp_targets = np.linspace(self.start_temp, target_temp, num=self.num_epochs + 1, endpoint=True)[1:]
        self.max_steps = 0
        print(
            f"Info: Temperature is adjusted {self.num_epochs} times"
            "in {self.ramp_targets[1] - self.ramp_targets[0]}K increments."
        )

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
            keys = ["idx", "indices", "index"]
            for key in keys:
                if hasattr(constraint, key):
                    val = np.array(getattr(constraint, key)).reshape(-1).tolist()
                    fixed_idx += val
                    has_keys = True
            if not has_keys:
                print(
                    (
                        "WARNING: velocity not set to zero for any atoms in constraint "
                        "%s; do not know how to find its fixed indices." % constraint
                    )
                )

        if not fixed_idx:
            return

        fixed_idx = np.array(list(set(fixed_idx)))
        vel = self.atoms.get_velocities()
        vel[fixed_idx] = 0
        self.atoms.set_velocities(vel)

    def step(self):
        # get current acceleration and velocity:
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        vel = self.atoms.get_velocities()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * accel

        # make full step in position
        x = self.atoms.get_positions() + vel_half * self.dt
        self.atoms.set_positions(x)

        # new accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        # make another half step in velocity
        vel = vel_half + 0.5 * self.dt * accel

        self.atoms.set_velocities(vel)
        self.remove_constrained_vel(self.atoms)

        return f

    def run(self):
        self.atoms.update_nbr_list()

        for ii in tqdm(range(self.num_epochs)):
            self.max_steps += self.nbr_update_period
            Dynamics.run(self)

            curr_temp = 2.0 * self.atoms.get_kinetic_energy() / (units.kB * self.activeDoF)
            curr_target = self.ramp_targets[ii]
            rescale_fac = np.sqrt(curr_target / curr_temp)
            new_vel = rescale_fac * self.atom.get_velocities()
            self.atoms.set_velocities(new_vel)

            self.atoms.update_nbr_list()
