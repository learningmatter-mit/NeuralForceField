import copy
import random
from datetime import datetime

import numpy as np
import torch
from ase.io.trajectory import Trajectory
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts
from nff.md.nms import nms_sample
from nff.md.nvt_ax import NoseHoover, NoseHooverChain
from nff.md.utils_ax import ZhuNakamuraLogger, atoms_to_nxyz, mol_dot, mol_norm
from nff.nn.utils import single_spec_nbrs
from nff.train import batch_detach, load_model
from nff.utils.constants import AMU_TO_AU, ASE_TO_FS, BOHR_RADIUS, FS_TO_AU, KB_AU, KCAL_TO_AU
from nff.utils.cuda import batch_to

HBAR = 1
OUT_FILE = "trj.csv"
LOG_FILE = "trj.log"
DEF_EXPLICIT_DIABAT = False
DEF_MAX_GAP_HOP = float("inf")
DEFAULT_SKIN = 1.0

METHOD_DIC = {"nosehoover": NoseHoover, "nosehooverchain": NoseHooverChain}


class ZhuNakamuraDynamics(ZhuNakamuraLogger):
    """
    Class for running Zhu-Nakamura surface-hopping dynamics. This method follows the description in
    Yu et. al, "Trajectory based nonadiabatic molecular dynamics without calculating nonadiabatic
    coupling in the avoided crossing case: Trans <-> cis photoisomerization in azobenzene ", Phys.
    Chem. Chem. Phys. 2014, doi: 10.1039/c4cp03498h.

    Attributes:

        atoms (ase.atoms.Atoms): atoms of the system
        dt (float): dynamics time-step
        max_time (float): maximum time of simulation
        time (float): current time
        num_states (int): total number of electronic states
        Natom (int): number of atoms
        out_file (str): name of output file with saved quantities
        log_file (str): name of file to log information about trajectory


        _time (float): current time in trajectory
        _positions (numpy.ndarray): array of atomic positions
        _velocities (numpy.ndarray): array of atomic velocities
        _forces (numpy.ndarray): array of shape (num_states, num_atoms, 3) for atomic forces,
            where num_states is the number of electronic states. The first dimension corresponds
            to which state the force is on, the second to which atom the force is on, and the third
            to which dimension the force is in.
        _energies (numpy.ndarray): array of shape (num_states). There is one energy for each state.
        _surf (int): current electronic state that the system is in
        _in_trj (bool): whether or not the current frame is "in the trajectory". The frame may not
            be in the trajectory in the following example. If an avoided crossing is found, and a
            hop occurs, then the last two frames are "removed" from the trajectory and replaced
            with a new frame. The new frame has the position of the second last frame, but a new
            surface and new re-scaled velocities. In this case the last two frames are not considered
            to be in the trajectory.
        _hopping_probabilities (list): A list of dictionaries with the Zhu a, b and p parameters.
            Each dictionary has information for hopping between different pairs of states.

        position_list (list): list of _positions at all past times in the trajectory
        velocity_list (list): list of _velocities at all past times in the trajectory
        force_list (list): list of _forces at all past times in the trajectory
        energy_list (list): list of _energies at all past times in the trajectory
        surf_list (list): list of _surf at all past times in the trajectory
        in_trj_list (list): list of _in_trj at all past times in the trajetory.
        time_list (list): list of times in trajectory
        hopping_probability_list (list): list of hopping probabilities at previous times

        diabatic_forces (numpy.ndarray): array of shape (2, num_atoms, 3) for the diabatic forces
            acting on a lower and upper diabatic state. If self.num_states > 2, the lower and upper
            diabatic states depend on which 2 of self.num_states are at an avoided crossing.
        diabatic_coupling (float): coupling strength between the upper and lower diabatic state.
        zhu_difference (float): Zhu difference parameter, used for calculating hopping probability
        zhu_product (float): Zhu product parameter, used for calculating hopping probability
        zhu_sign (int): Zhu sign parameter (+/- 1), used for calculating hopping probability
        n_vector (numpy.ndarray): Zhu n-vector, of shape (num_atoms, 3), used for calculating
            hopping probability
        v_parallel (numpy.ndarray): Component of the velocity parallel to the hopping direction. Has
            shape (num_atoms), and is used for calculating hopping probability.
        ke_parallel (float): Kinetic energy associated with v_parallel.
        ke (float): Total kinetic energy
        hopping_probabilities (list): A list of dictionaries with the Zhu a, b and p parameters.
            Each dictionary has information for hopping between different pairs of states.

    Properties:

        positions: returns self._positions. Updating positions updates self._positions,
            self.positions_list, and positions of self.atoms.
        velocities: returns self._velocities. Updating positions updates self._velocities,
            self.velocities_list and velocities of self.atoms.
        forces: returns self._forces. Updating forces updates self._forces, self.forces_list
            and forces of self.atoms.
        energies: returns self._energies. Updating energies updates self._energies,
            self.energy_list and energies of self.atoms.
        surf: returns self._surf. Updating surf updates self._surf and self.surf_list.
        in_trj: returns self._in_trj. Updating in_trj updates self._in_trj and self.
        time: returns self._time. Updating time updates self.time_list.
        hopping_probabilities: returns self._hopping_probabilities. Updating hopping_probabilities
            updates self.hopping_probability_list
    """

    def __init__(
        self,
        atoms,
        timestep,
        max_time,
        explicit_diabat=DEF_EXPLICIT_DIABAT,
        max_gap_hop=DEF_MAX_GAP_HOP,
        initial_time=0.0,
        initial_surf=1,
        num_states=2,
        out_file=OUT_FILE,
        log_file=LOG_FILE,
        save_period=None,
        **kwargs,
    ):
        """
        Initializes a ZhuNakamura instance.

        Args:
            atoms (ase.atoms.Atoms): atoms of the system
            timestep (float): timestep for the dynamics, in femtoseconds
            initial_time (float): initial time for the dynamics
            max_time (float): total time of simulation
            initial_surf (int): initial electronic state for the dynamics. Note that, as always, we use
                Python numbering, so that initial_surf = 1 means you start on the first excited state,
                and initial_surf = 0 means you start on the ground state.
            num_states (int): number of total electronic states
            trajectory ():
            logfile ():
            loginterval ():

        Returns:
            None

        """

        self.atoms = atoms
        self.dt = timestep * FS_TO_AU
        self.max_time = max_time * FS_TO_AU
        self.num_states = num_states
        self.Natom = len(atoms)
        self.max_gap_hop = max_gap_hop
        self.explicit_diabat = explicit_diabat
        self.diabat_ens = None
        self.diabat_forces = None

        self.out_file = out_file
        self.log_file = log_file
        self.save_period = save_period if save_period is not None else self.dt / FS_TO_AU
        self.setup_logging()

        # everything in a.u. other than positions (which are in angstrom)
        self._positions = atoms.get_positions()
        self._velocities = atoms.get_velocities() / BOHR_RADIUS / (ASE_TO_FS * FS_TO_AU)

        self._forces = None
        self._energies = None
        self._surf = initial_surf
        self._in_trj = True
        self._time = initial_time * FS_TO_AU
        self._hopping_probabilities = []

        self.position_list = [self._positions]
        self.velocity_list = [self._velocities]
        self.force_list = None
        self.energy_list = None
        self.surf_list = [self._surf]
        self.in_trj_list = [self._in_trj]
        self.time_list = [self._time]
        self.hopping_probability_list = [self._hopping_probabilities]
        self.old_accel = None

        # Initialize Zhu-Nakamura quantities
        self.diabatic_forces = np.array([])
        self.diabatic_coupling = 0.0
        self.zhu_difference = 0.0
        self.zhu_product = 0.0
        self.zhu_sign = 0
        self.n_vector = np.array([])
        self.v_parallel = np.array([])
        self.ke_parallel = 0.0
        self.ke = 0.0

        save_keys = [
            "position_list",
            "velocity_list",
            "force_list",
            "energy_list",
            "surf_list",
            "in_trj_list",
            "hopping_probability_list",
            "time_list",
        ]

        super().__init__(save_keys=save_keys, **self.__dict__)

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def forces(self):
        return self._forces

    @property
    def energies(self):
        return self._energies

    @property
    def surf(self):
        return self._surf

    @property
    def in_trj(self):
        return self._in_trj

    @property
    def time(self):
        return self._time

    @property
    def hopping_probabilities(self):
        return self._hopping_probabilities

    @positions.setter
    def positions(self, value):
        self._positions = value
        # add positions to position_list
        self.position_list.append(value)
        # update the positions of self.atoms
        self.atoms.set_positions(value)

    @velocities.setter
    def velocities(self, value):
        """
        Automatically update quantities associated with velocities when changing the velocities.
        Args:
            value (numpy.ndarray): new array of velocities
        Returns:
            None
        """

        self._velocities = value
        # add velocities to velocity_list
        self.velocity_list.append(value)
        # update the velocities of self.atoms
        self.atoms.set_velocities(value)

    @forces.setter
    def forces(self, value):
        """
        Automatically update quantities associated with forces when changing the forces.
        Args:
            value (numpy.ndarray): new array of forces
        Returns:
            None
        """

        self._forces = value
        # add forces to force_list
        if hasattr(self.force_list, "__iter__"):
            self.force_list.append(value)
        else:
            self.force_list = [value]

    @energies.setter
    def energies(self, value):
        """
        Automatically update quantities associated with energies when changing the forces.
        Args:
            value (numpy.ndarray): new array of energies
        Returns:
            None
        """

        self._energies = value
        # add energies to energy_list

        if hasattr(self.energy_list, "__iter__"):
            self.energy_list.append(value)
        else:
            self.energy_list = [value]

    @surf.setter
    def surf(self, value):
        """
        Automatically update quantities associated with surf when changing the surface.
        Args:
            value (int): new surface
        Returns:
            None
        """

        self._surf = value
        # add surf to surf_list
        self.surf_list.append(value)

    @in_trj.setter
    def in_trj(self, value):
        """
        Automatically update quantities associated with in_trj when changing in_trj.
        Args:
            value (bool): whether or not the new frame is in the trajectory.
        Returns:
            None
        """
        self._in_trj = value
        # add in_trj to in_trj_list
        self.in_trj_list.append(value)

    @time.setter
    def time(self, value):
        """
        Automatically update quantities associated with time when changing time.
        Args:
            value (float): new time
        Returns:
            None
        """

        self._time = value
        # add time to time_list
        self.time_list.append(value)

    @hopping_probabilities.setter
    def hopping_probabilities(self, value):
        """
        Automatically update quantities associated with hopping probabilities when changing it.
        Args:
            value (list): new hopping probabilities
        Returns:
            None
        """

        self._hopping_probabilities = value
        self.hopping_probability_list.append(value)

    def update_forces(self):
        """
        Update self.forces by get_forces() on self.atoms
        """
        self.forces = self.atoms.get_forces()

    def update_energies(self):
        """
        Update self.energies by get_potential_energy() on self.atoms
        """
        self.energies = self.atoms.get_potential_energy()

    def get_masses(self):
        """
        Get masses of system atoms.
        Returns:
            self.atoms.get_masses() (numpy.ndarray): masses
        """
        return self.atoms.get_masses() * AMU_TO_AU

    def get_accel(self):
        """
        Get current acceleration of atoms
        Returns:
            accel (nump.ndarray): acceleration
        """

        # the force is force acting on the current state

        force = self.forces[self.surf]
        accel = force / self.get_masses().reshape(-1, 1)
        return accel

    def position_step(self):
        # get current acceleration and velocity
        accel = self.get_accel()
        self.old_accel = accel

        # take a step for the positions
        # positions are in Angstrom so they must be properly converted
        # Note also that we don't use += here, because that causes problems
        # with setters.

        self.positions = self.positions + (self.velocities * self.dt + 1 / 2 * accel * self.dt**2) * BOHR_RADIUS

    def velocity_step(self, do_log=True):
        new_accel = self.get_accel()
        self.velocities = self.velocities + 1 / 2 * (new_accel + self.old_accel) * self.dt
        # assume the current frame is in the trajectory until
        # finding out otherwise
        self.in_trj = True
        # update surf (which also appends to surf_list)
        self.surf = self.surf
        self.time = self.time + self.dt

        step = int(self.time / self.dt)
        rel_ens = ", ".join(((self.energies - self.energies[0]) * 27.2).reshape(-1).astype("str").tolist())

        if do_log:
            self.log(f"Completed step {step}. Currently in state {self.surf}.")
            self.log(f"Relative energies are {rel_ens} eV")

    def md_step(self):
        """
        Take a regular molecular dynamics step on the current surface.
        """

        self.position_step()
        # get forces and energies at new positions
        self.update_forces()
        self.update_energies()
        self.velocity_step()

    def check_crossing(self):
        """Check if we're at an avoided crossing by seeing if the energy gap
         was at a minimum  in the last step.
        Args:
            None
        Returns:
            at_crossing (bool): whether we're at an avoided crossing for any combination of states
            new_surfs (list): list of surfaces that are at an avoided crossing with the current
                surface.
        """

        new_surfs = []
        at_crossing = False

        # if we've taken less than three steps, we can't check if we're at
        # an avoided crossing
        if len(self.energy_list) < 3 or len(self.surf_list) < 3:
            return at_crossing, new_surfs

        # all of the past three steps must be in the trajectory. This stops us
        # from trying to re-hop after we've already hopped at an avoided
        # crossing. If a hop has already happened, then you don't try it again
        #  at the same position.

        if not all(is_in_trj for is_in_trj in self.in_trj_list[-3:]):
            return at_crossing, new_surfs

        # loop through states other than self.surf and see if they're
        # at an avoided crossing
        for i in range(self.num_states):
            if i == self.surf:
                continue
            # list of energy gaps
            gaps = [abs(energies[i] - energies[self.surf]) for energies in self.energy_list[-3:]]
            # whether or not the middle gap is the smallest of the three
            gap_min = gaps[0] > gaps[1] and gaps[2] > gaps[1]
            if gap_min:
                new_surfs.append(i)
                at_crossing = True

        return at_crossing, new_surfs

    def get_diabat_engrads(self, lower_state, upper_state):
        # update diabatic forces. Start with the r_{ij} parameters
        # from the ZN paper units for r_{ij} don't matter since
        # they only get called in ratios

        r_20 = self.position_list[-1] - self.position_list[-3]
        r_10 = self.position_list[-2] - self.position_list[-3]
        r_12 = self.position_list[-2] - self.position_list[-1]

        # diabatic forecs on the lower state
        lower_diabatic_forces = (
            -(-self.force_list[-1][lower_state] * r_10 + self.force_list[-3][upper_state] * r_12) / r_20
        )
        # diabatic forces on the upper state
        upper_diabatic_forces = (
            -(-self.force_list[-1][upper_state] * r_10 + self.force_list[-3][lower_state] * r_12) / r_20
        )

        # array of forces on the lower and upper diabatic states
        diabatic_forces = np.append([lower_diabatic_forces], [upper_diabatic_forces], axis=0)

        # update diabatic coupling
        diabatic_coupling = (self.energy_list[-2][upper_state].item() - self.energy_list[-2][lower_state].item()) / 2

        return diabatic_forces, diabatic_coupling

    def update_diabatic_quants(self, lower_state, upper_state):
        """
        Update diabatic quantities at an avoided crossing.
        Args:
            lower_state (int): index of lower electronic state at crossing
            upper_state (int): index of upper electronic stte at crossing
        Returns:
            None

        """

        if self.explicit_diabat:
            if self.diabat_ens is None:
                raise Exception("Diabatic quantities haven't been updated")
            self.diabatic_coupling = abs(self.diabat_ens[lower_state, upper_state])
            state_array = np.array([lower_state, upper_state])
            self.diabatic_forces = self.diabat_forces[state_array, :]

        else:
            d_forces, d_coupling = self.get_diabat_engrads(lower_state=lower_state, upper_state=upper_state)

            self.diabatic_forces = d_forces
            self.diabatic_coupling = d_coupling

        # update Zhu difference parameter
        norm_vec = mol_norm(self.diabatic_forces[1] - self.diabatic_forces[0])
        self.zhu_difference = np.sum(norm_vec**2 / self.get_masses()) ** 0.5

        # update Zhu product parameter and the Zhu sign parameter
        prods = self.diabatic_forces[0] * self.diabatic_forces[1]
        inner = np.sum(prods / self.get_masses().reshape(-1, 1))
        self.zhu_product = abs(inner) ** 0.5
        self.zhu_sign = int(np.sign(inner))

        # get parallel component of velocity and the associated KE
        # First normalize s-vector to give n-vector
        s = (self.diabatic_forces[1] - self.diabatic_forces[0]) / self.get_masses().reshape(-1, 1) ** 0.5
        self.n_vector = s / mol_norm(s).reshape(-1, 1)

        # Then get ke's
        self.v_parallel = mol_dot(self.velocity_list[-2], self.n_vector)
        self.ke_parallel = np.sum(self.get_masses() * (self.v_parallel**2) / 2)
        self.ke = np.sum(self.get_masses() * mol_norm(self.velocity_list[-2]) ** 2 / 2)

    def rescale_v(self, old_surf, new_surf):
        """
        Re-scale the velocity after a hop.
        Args:
            old_surf (int): old surface
            new_surf (int): new surface
        Returns:
            None
        """

        # the energy to consider is actually the energy at the crossing point,
        # which is not the current energy but the energy one step before
        energy = self.energy_list[-2]
        # the component of v parallel to the hopping direction
        v_par_vec = self.n_vector * (self.v_parallel).reshape(-1, 1)
        # the scaling factor for the velocities

        scale_arg = ((energy[old_surf] + (self.ke_parallel)) - energy[new_surf]) / (self.ke_parallel)

        if scale_arg < 0 or np.isnan(scale_arg):
            return "err"

        scale = scale_arg**0.5
        velocities = scale * v_par_vec + (self.velocity_list[-2] - v_par_vec)

        if np.isnan(velocities).any():
            return "err"
        self.velocities = velocities
        return None

    def update_probabilities(self):
        """
        Update the Zhu a, b and p probabilities.
        """

        hopping_probabilities = []
        at_crossing, new_surfs = self.check_crossing()
        # if we're not at a crossing, then the hopping probabilities
        # shouldn't be considered
        if not at_crossing:
            self.hopping_probabilities = hopping_probabilities
            return

        # if the molecule's exploded then move on
        if "nan" in self.positions.astype("str") or "nan" in self.forces.astype("str"):
            self.hopping_probabilities = hopping_probabilities
            return

        for new_surf in new_surfs:
            # get the upper and lower state by sorting the current
            # surface and the new one
            lower_state, upper_state = sorted((self.surf, new_surf))

            # If requested, only consider a hop if the gap is below
            # a certain value. Don't just get the gap as 2 * the
            # diabatic coupling, because that's not the case if
            # we use explicit diabatic states

            gap = abs(self.energy_list[-2][upper_state].item() - self.energy_list[-2][lower_state].item())

            if gap > self.max_gap_hop:
                hopping_probabilities.append({"zhu_a": 0, "zhu_b": 0, "zhu_p": 0, "new_surf": new_surf})
                continue

            # another place it might fail with nan
            try:
                self.update_diabatic_quants(lower_state, upper_state)
            except ValueError:
                return

            # use context manager to ignore any divide by 0's
            with np.errstate(divide="ignore", invalid="ignore"):
                # calculate the zhu a parameter
                a_numerator = HBAR**2 / 2 * self.zhu_product * self.zhu_difference
                a_denominator = (2 * self.diabatic_coupling) ** 3
                zhu_a = np.nan_to_num(np.divide(a_numerator, a_denominator) ** 0.5)

                # calculate the zhu b parameter, starting with Et and Ex
                et = self.ke_parallel + self.energy_list[-2][self.surf].item()
                ex = (self.energy_list[-2][upper_state].item() + self.energy_list[-2][lower_state].item()) / 2
                b_numerator = (et - ex) * self.zhu_difference / self.zhu_product
                b_denominator = 2 * self.diabatic_coupling
                zhu_b = np.nan_to_num(np.divide(b_numerator, b_denominator) ** 0.5)

                # calculating the hopping probability
                zhu_p = np.nan_to_num(
                    np.exp(
                        -np.pi / 4 / zhu_a * (2 / (zhu_b**2 + (abs((zhu_b**4) + (self.zhu_sign) * 1.0)) ** 0.5)) ** 0.5
                    )
                )

                # add this info to the list of hopping probabilities

                hopping_probabilities.append({"zhu_a": zhu_a, "zhu_b": zhu_b, "zhu_p": zhu_p, "new_surf": new_surf})

        self.hopping_probabilities = hopping_probabilities

    def should_hop(self, zhu_p):
        """
        Decide whether or not to hop based on the zhu a, b and p parameters.
        Args:
            zhu_a (float): Zhu a parameter
            zhu_b (float): Zhu b parameter
            zhu_p (float): hopping probability
        Returns:
            will_hop (bool): whether or not to hop

        """

        rnd = np.random.rand()
        will_hop = zhu_p > rnd

        return will_hop

    def hop(self, new_surf):
        """
        Hop from the current surface to a new surface at an avoided crossing.
        Args:
            new_surf (int): index of new surface
        Returns:
            None
        """

        # re-scale the velocity
        out = self.rescale_v(old_surf=self.surf, new_surf=new_surf)
        if out == "err":
            return out

        # change the surface
        self.surf = new_surf

        # reset to the second last position for positions, energies, and forces
        self.positions = self.position_list[-2]
        self.energies = self.energy_list[-2]
        self.forces = self.force_list[-2]

        # set the frame to be in the trajectory, but the previous
        # two frames to be out of the trajectory
        self.in_trj = True
        self.in_trj_list[-2] = False
        self.in_trj_list[-3] = False

        # add a new empty hopping probability list
        self.hopping_probabilities = []
        self.time = self.time - self.dt
        self.modify_save()
        return None

    def full_step(self, compute_internal_forces=True, do_log=True):
        """

        Take a time step.

        """

        if compute_internal_forces:
            self.md_step()
        # update the hopping probabilities
        self.update_probabilities()

        # randomly order the self.hopping_probabilities list.
        # If, for some reason, two sets of states
        # are both at an avoided crossing, then we'll first
        # try to hop between the first set of states.
        # If this fails then we'll try to hop between the second
        # set of states. To avoid biasing in
        # the direction of one hop vs. another, we randomly shuffle
        # the order of self.hopping_probabilities
        # each time.

        random.shuffle(self.hopping_probabilities)

        # loop through sets of states to hop between

        for probability_dic in self.hopping_probabilities:
            zhu_p = probability_dic["zhu_p"]
            new_surf = probability_dic["new_surf"]
            old_surf = copy.deepcopy(self.surf)

            if do_log:
                self.log(f"Attempting hop from state {old_surf} to state " f"{new_surf}. Probability is {zhu_p}.")

            # decide whether or not to hop based on Zhu a, b, and p
            will_hop = self.should_hop(zhu_p)

            # hop and end loop if will_hop == True
            if will_hop:
                out = self.hop(new_surf)
                if out != "err":
                    if do_log:
                        self.log(f"Hopped from state {old_surf} " f"to state {new_surf}.")
                    return
            else:
                if do_log:
                    self.log(f"Did not hop from state {old_surf} " f"to state {new_surf}.")

    def run(self):
        # save intitial conditions

        self.update_energies()
        self.update_forces()

        self.save()
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rel_ens = ", ".join(((self.energies - self.energies[0]) * 27.2).reshape(-1).astype("str").tolist())

        self.log(f"Beginning surface hopping at {time_str}.")
        self.log(f"Relative energies are {rel_ens} eV")

        while self.time < self.max_time:
            self.step()
            self.save()

        # self.output_to_json()
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"Surface hopping completed normally at {time_str}.")


class NoseHooverZN(ZhuNakamuraDynamics):
    def __init__(self, temperature, ttime, **kwargs):
        ZhuNakamuraDynamics.__init__(self, **kwargs)
        self.zeta = 0.0
        self.ttime = ttime
        self.temp = temperature * KB_AU
        n_atom = len(self.atoms)
        self.n_dof = 3.0 * n_atom - 6
        self.targe_ekin = 0.5 * self.n_dof * self.temp
        self.Q = self.n_dof * self.temp * (self.ttime * self.dt) ** 2

    def get_kinetic_energy(self, vel=None):
        if vel is None:
            vel = self.velocities
        ke = np.sum(self.get_masses() * mol_norm(vel) ** 2 / 2)
        return ke

    def position_step(self):
        # get current acceleration and velocity
        accel = self.get_accel()
        self.old_accel = accel

        delta_pos = (self.velocities * self.dt + (accel - self.zeta * self.velocities) * 0.5 * self.dt**2) * BOHR_RADIUS
        self.positions = self.positions + delta_pos

    def velocity_step(self, do_log=True):
        # NVT stuff

        # ke before half velocity step
        ke_0 = self.get_kinetic_energy()

        # make a half step in velocity
        # (don't update v yet because it will mess up the "append to
        # velocity list" part of the velocity setter)

        v_half = self.velocities + 0.5 * self.dt * (self.old_accel - self.zeta * self.velocities)

        # make a half step in zeta
        z_half = self.zeta + 0.5 * self.dt / self.Q * (ke_0 - self.targe_ekin)

        # make another half step in zeta
        ke_1 = self.get_kinetic_energy(v_half)
        self.zeta = z_half + 0.5 * self.dt / self.Q * (ke_1 - self.targe_ekin)

        # make another half step in velocity
        new_accel = self.get_accel()
        self.velocities = (v_half + 0.5 * self.dt * new_accel) / (1 + 0.5 * self.dt * self.zeta)

        # temp = self.get_kinetic_energy() / (1 / 2 * self.n_dof) / KB_AU
        # print("Temperature = %.2f K" % temp)

        # ZN stuff
        # assume the current frame is in the trajectory until
        # finding out otherwise
        self.in_trj = True
        # update surf (which also appends to surf_list)
        self.surf = self.surf
        self.time = self.time + self.dt

        step = int(self.time / self.dt)
        rel_ens = ", ".join(((self.energies - self.energies[0]) * 27.2).reshape(-1).astype("str").tolist())

        if do_log:
            self.log(f"Completed step {step}. Currently in state {self.surf}.")
            self.log(f"Relative energies are {rel_ens} eV")


class BatchedZhuNakamura:
    """
    A class for running several Zhu Nakamura trajectories at once. This is done by taking a half
    step for each trajectory, combining all the xyz's into a dataset and batching it for the
    network, and then de-batching to put the forces and energies
    back in the trajectories.

    Attributes:
        num_trj (int): number of concurrent trajectories
        zhu_trjs (list): list of ZhuNakamura instances
        max_time (float): maximum simulation time
        energy_keys (list): names of outputted energies
        grad_keys (list): names of outputted gradient keys
        props (dict): dataset properties
        nbr_update_period (float): how often to update the neighbor list
        device (int): GPU device
        model (torch.nn): neural network model
        batch_size (int): size of batches to be fed into network
        cutoff (float): neighbor list cutoff in schnet
        cutoff_skin (float): extra amount of distance to add to cutoff
            to deal with atoms becoming neighbors between neighbor list
            updates

    """

    def __init__(
        self, atoms_list, props, batched_params, zhu_params, modelparams=None, model_type=None, needs_angles=False
    ):
        """
        Initialize.
        Args:
            atoms_list (list): list of ASE atom objects
            props (dict): dictionary of dataset props
            batched_params (dict): parameters related to the batching process
            zhu_params (dict): parameters related to Zhu Nakamura
        """

        self.num_trj = batched_params["num_trj"]
        self.zhu_trjs = self.make_zhu_trjs(atoms_list, zhu_params)
        self.explicit_diabat = zhu_params.get("explicit_diabat", DEF_EXPLICIT_DIABAT)
        self.max_time = self.zhu_trjs[0].max_time
        if "en_key_list" not in zhu_params:
            self.energy_keys = [f"energy_{i}" for i in range(self.zhu_trjs[0].num_states)]
        else:
            self.energy_keys = zhu_params["en_key_list"]
        if len(self.energy_keys) != self.zhu_trjs[0].num_states:
            raise ValueError

        self.grad_keys = [f"{key}_grad" for key in self.energy_keys]

        self.props = self.duplicate_props(props)
        self.nbr_update_period = batched_params["nbr_update_period"]
        self.device = batched_params["device"]
        self.model = load_model(batched_params["weight_path"], params=modelparams, model_type=model_type)
        self.model.eval()
        self.model.to(self.device)

        self.batch_size = batched_params["batch_size"]
        self.cutoff = batched_params["cutoff"]
        self.cutoff_skin = batched_params.get("cutoff_skin", DEFAULT_SKIN)
        self.needs_angles = needs_angles

        # for saving at intervals
        self.save_period = min([trj.save_period for trj in self.zhu_trjs])
        self.dt = min([trj.dt for trj in self.zhu_trjs])

    def make_zhu_trjs(self, atoms_list, zhu_params):
        """
        Instantiate the Zhu Nakamura objects.
        Args:
            atoms_list (list): list of ASE atom objects
            zhu_params (dict): parameters related to Zhu Nakamura
        Returns:
            zhu_trjs (list): list of ZhuNakamura trajectory objects
        """

        assert len(atoms_list) == self.num_trj

        # base names for the output and log files
        base_out_name = zhu_params.get("out_file", OUT_FILE).split(".csv")[0]
        base_log_name = zhu_params.get("log_file", LOG_FILE).split(".log")[0]
        zhu_trjs = []

        for i, atoms in enumerate(atoms_list):
            these_params = copy.deepcopy(zhu_params)
            these_params["out_file"] = f"{base_out_name}_{i}.csv"
            these_params["log_file"] = f"{base_log_name}_{i}.log"

            thermostat = these_params.get("thermostat", "none").lower()
            class_dic = {"none": ZhuNakamuraDynamics, "nosehoover": NoseHooverZN}
            zn_class = class_dic[thermostat]
            zhu_trjs.append(zn_class(atoms=atoms, **these_params))

        return zhu_trjs

    def duplicate_props(self, props):
        """
        Duplicate properties, once for each trajectory.
        Args:
            props (dict): dictionary of dataset props
        Returns:
            new_props (dict): dictionary updated for each trajectory
        """

        new_props = dict()
        for key, val in props.items():
            if isinstance(val, list):
                new_props[key] = val * self.num_trj
            elif hasattr(val, "tolist"):
                typ = type(val)
                new_props[key] = typ((val.tolist()) * self.num_trj)
            else:
                raise Exception

        new_props.update({key: None for key in [*self.energy_keys, *self.grad_keys]})
        new_props["num_atoms"] = new_props["num_atoms"].long()

        return new_props

    def update_energies_forces(self, trjs, get_new_neighbors):
        """
        Update the energies and forces for the molecules of each trajectory.
        Args:
            trjs (list): list of trajectories
            get_new_neighbors (bool): whether or not to update the neighbor list
        Returns:
            None
        """

        nxyz_data = [torch.Tensor(atoms_to_nxyz(trj.atoms)) for trj in trjs]
        props = {"nxyz": nxyz_data, "nbr_list": self.props["nbr_list"], "num_atoms": self.props["num_atoms"]}
        dataset = Dataset(props=props, units="kcal/mol", check_props=False)

        if get_new_neighbors:
            # can stack the nxyz's and generate the neighbor list
            # accordingly because all geoms correspond to the
            # same molecule
            nbrs = single_spec_nbrs(
                dset=dataset, cutoff=(self.cutoff + self.cutoff_skin), device=self.device, directed=True
            )
            dataset.props["nbr_list"] = nbrs

            # dataset.generate_neighbor_list(self.cutoff)

            if self.needs_angles:
                dataset.generate_angle_list()

        dataset.props["num_atoms"] = dataset.props["num_atoms"].long()
        self.props = dataset.props
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_dicts)

        for i, batch in enumerate(loader):
            batch = batch_to(batch, self.device)
            results = self.model(batch)
            results = batch_detach(results)

            for key in self.grad_keys:
                N = batch["num_atoms"].cpu().detach().numpy().tolist()
                results[key] = torch.split(results[key], N)

            current_trj = i * self.batch_size

            for j, trj in enumerate(trjs[current_trj : current_trj + self.batch_size]):
                energies = []
                forces = []
                for key in self.energy_keys:
                    energy = (results[key][j].item()) * KCAL_TO_AU["energy"]

                    force = (
                        ((-results[key + "_grad"][j]).detach().cpu().numpy())
                        * KCAL_TO_AU["energy"]
                        * KCAL_TO_AU["_grad"]
                    )

                    energies.append(energy)
                    forces.append(force)

                trj.energies = np.array(energies)
                trj.forces = np.array(forces)

    def add_diabat_forces(self):
        diabat_trjs = []
        diabat_idx = []

        for i, trj in enumerate(self.zhu_trjs):
            at_crossing, _ = trj.check_crossing()
            if at_crossing:
                diabat_trjs.append(trj)
                diabat_idx.append(i)
            else:
                # reset to None to catch any silent errors of
                # reusing the old diabatic forces
                trj.diabat_ens = None
                trj.diabat_forces = None

        # create a dataset and limit to only the trajectories you
        # care about

        dataset = Dataset(props=self.props.copy(), units="kcal/mol", check_props=False)
        diabat_idx = torch.LongTensor(diabat_idx)
        dataset.change_idx(diabat_idx)

        # get positions at previous time step for all trajectories at
        # crossings

        xyz_data = [trj.position_list[-2] for trj in diabat_trjs]
        for i, xyz in enumerate(xyz_data):
            z_arr = diabat_trjs[i].atoms.get_atomic_numbers()
            nxyz = np.concatenate([z_arr.reshape(-1, 1), xyz], axis=-1)
            dataset.props["nxyz"][i] = torch.Tensor(nxyz)

        # technically not generating neighbors here isn't totally consistent,
        # because it's possible that neighbors were generated at the subsequent
        # step in `update_energies_forces`, and you're using those neighbors
        # now, whereas they really weren't used in the original calculation of
        # the forces at this step. But assuming the neighbor list is getting
        # updated frequently enough that this doesn't affect the engrads too
        # much, we shouldn't have to worry about it

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_dicts)

        for i, batch in enumerate(loader):
            batch = batch_to(batch, self.device)
            diabat_keys = np.array(self.model.diabatic_readout.diabat_keys)
            diag_diabats = diabat_keys.diagonal()
            extra_grads = [f"{key}_grad" for key in diag_diabats]
            results = batch_detach(self.model(batch, extra_grads=extra_grads))

            for key in [*self.grad_keys, *extra_grads]:
                N = batch["num_atoms"].cpu().detach().numpy().tolist()
                results[key] = torch.split(results[key], N)

            current_trj = i * self.batch_size
            end_trj = current_trj + self.batch_size

            for j, trj in enumerate(diabat_trjs[current_trj:end_trj]):
                num_states = diabat_keys.shape[0]

                # store the diabatic energies as a matrix
                diabat_ens = np.zeros((num_states, num_states))

                # only store the diagonal diabatic forces
                diabat_forces = np.zeros((num_states, N[j], 3))

                for k in range(num_states):
                    for m in range(num_states):
                        d_key = diabat_keys[k, m]
                        diabat_en_kcal = results[d_key][j].item()
                        diabat_en_au = diabat_en_kcal * KCAL_TO_AU["energy"]

                        diabat_ens[k, m] = diabat_en_au

                        if k == m:
                            diabat_force_kcal = -(results[f"{d_key}_grad"][j].detach().cpu().numpy())
                            diabat_force_au = diabat_force_kcal * KCAL_TO_AU["energy"] * KCAL_TO_AU["_grad"]
                            diabat_forces[k, :] = diabat_force_au

                trj.diabat_ens = diabat_ens
                trj.diabat_forces = diabat_forces

    def single_pos_step(self, i):
        self.zhu_trjs[i].position_step()

    def save_par(self, i):
        trj = self.zhu_trjs[i]
        # trj.velocity_step()
        # trj.full_step(compute_internal_forces=False)
        # print(trj.time)
        if trj.time < self.max_time:
            trj.save()

    def step(self, get_new_neighbors, do_save=True):
        """
        Take a step for each trajectory
        Args:
            get_new_neighbors (bool): whether to update the neighbor list
        Returns:
            None
        """

        for trj in self.zhu_trjs:
            # take a position step based on previous energies and forces
            trj.position_step()

        # run position steps in parallel - for some reason a single process
        # is the most efficient and is still much faster than a for-loop
        # when it comes to saving csvs

        # pool.imap_unordered(self.single_pos_step, range(num_trjs))

        # update the energies and forces
        self.update_energies_forces(trjs=self.zhu_trjs, get_new_neighbors=get_new_neighbors)

        for trj in self.zhu_trjs:
            # take a velocity step
            trj.velocity_step(do_log=do_save)

        if self.explicit_diabat:
            self.add_diabat_forces()

        for trj in self.zhu_trjs:
            # take a "full_step" with compute_internal_forces=False,
            # which just amounts to checking if you're at a crossing and
            # potentially hopping
            trj.full_step(compute_internal_forces=False, do_log=do_save)

        for trj in self.zhu_trjs:
            if trj.time < self.max_time and do_save:
                trj.save()

        # pool = Pool(processes=1)
        # num_trjs = len(self.zhu_trjs)
        # pool.imap_unordered(self.save_par, range(1))
        # pool.close()

        # multi_pool = Pool(processes=5)
        # multi_pool.map(self.save_par, range(1))
        # multi_pool.close()
        # multi_pool.join()

        # q = mp.Queue()
        # p = mp.Process(target=self.save_par, args=range(1))
        # p.start()
        # print(q.get())
        # p.join()

    def run(self):
        """
        Run all the trajectories
        """

        # initial energy and force calculation to get things started
        self.update_energies_forces(trjs=self.zhu_trjs, get_new_neighbors=True)
        complete = False
        num_steps = 0
        save_steps = int(self.save_period / (self.dt / FS_TO_AU))

        while not complete:
            get_new_neighbors = np.mod(num_steps, self.nbr_update_period) == 0
            do_save = np.mod(num_steps, save_steps) == 0

            self.step(get_new_neighbors=get_new_neighbors, do_save=do_save)

            if do_save:
                print(f"Completed step {num_steps}")

            complete = all(trj.time >= self.max_time for trj in self.zhu_trjs)
            num_steps += 1

        print("Neural ZN terminated normally.")

        # for trj in self.zhu_trjs:
        #     trj.output_to_json()


class CombinedZhuNakamura:
    """
    Class for combining an initial ground state MD simulation with BatchedZhuNakamura.
    Attributes:
        ground_dynamics: trajectory on the ground state
        ground_savefile (str) : name of output file from ground state trajectory
        equil_time (float): length of time to let the system equilibrate on the ground state
            before sampling geometries for a subsequent Zhu-Nakamura run
        num_trj (int): number of excited state trajectories to run in parallel
        zhu_params (dict): parameters for Zhu-Nakamura run
        batched_params (dict): parameters for batching Zhu Nakamura
        props (dict): dataset props
        ground_params (dict): parameters for ground state MD
    """

    def __init__(
        self,
        atoms,
        zhu_params,
        batched_params,
        ground_params,
        props,
        modelparams=None,
        model_type=None,
        needs_angles=False,
    ):
        """
        Initialize:
            atoms: ase Atoms objects
            zhu_params: see above
            batched_params: see above
            ground_params: see above
            props: see above
        """

        ase_ground_params = copy.deepcopy(ground_params)
        # the Dynamics classes we've made automatically convert to ASE units
        # ase_ground_params["max_time"] *= FS_TO_ASE
        # ase_ground_params["timestep"] *= FS_TO_ASE
        ase_ground_params["trajectory"] = ground_params.get("savefile")
        # ase_ground_params["temperature"] =  ground_params["temperature"]*KB_EV

        self.nms = ase_ground_params.get("nms")
        if not self.nms:
            method = METHOD_DIC[ase_ground_params["thermostat"]]
            self.ground_dynamics = method(atoms, **ase_ground_params)

        self.ground_savefile = ground_params.get("savefile")
        self.equil_time = ground_params.get("equil_time")
        self.num_trj = batched_params["num_trj"]
        self.zhu_params = zhu_params
        self.batched_params = batched_params
        self.props = props

        self.ground_params = ground_params

        self.modelparams = modelparams
        self.model_type = model_type
        self.needs_angles = needs_angles

    def sample_ground_geoms(self):
        """
        Run a ground state trajectory and extract starting geometries and velocities for each
        Zhu Nakamura trajectory.
        Args:
            None
        Returns:
            actual_states (list): list of atoms objects extracted from the trajectories.
        """

        if self.nms:
            temp = self.ground_params.get("temperature", self.ground_params.get("T_init"))
            actual_states = nms_sample(
                params=self.ground_params,
                classical=self.ground_params["classical"],
                num_samples=self.num_trj,
                kt=25.7 / 1000 / 27.2 * temp / 300,
                hb=1,
            )

            return actual_states

        steps = int(self.ground_params["max_time"] / self.ground_params["timestep"])
        equil_steps = int(self.ground_params["equil_time"] / self.ground_params["timestep"])

        self.ground_dynamics.run(steps=steps)
        trj = Trajectory(self.ground_savefile)

        loginterval = self.ground_params.get("loginterval", 1)
        logged_equil = int(equil_steps / loginterval)
        possible_states = [trj[index] for index in range(logged_equil, len(trj))]
        random_indices = random.sample(range(len(possible_states)), self.num_trj)
        actual_states = [possible_states[index] for index in random_indices]

        return actual_states

    def run(self):
        """
        Run a ground state trajectory followed by a set of parallel Zhu Nakamura trajectories.
        """

        set_start_method("spawn")

        atoms_list = self.sample_ground_geoms()

        # print(atoms_list[0].get_kinetic_energy())
        # print(atoms_list[0].get_positions())
        # print(atoms_list[0].get_velocities())

        # mp.set_start_method('spawn')

        batched_zn = BatchedZhuNakamura(
            atoms_list=atoms_list,
            props=self.props,
            batched_params=self.batched_params,
            zhu_params=self.zhu_params,
            modelparams=self.modelparams,
            model_type=self.model_type,
            needs_angles=self.needs_angles,
        )

        batched_zn.run()
