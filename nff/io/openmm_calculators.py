from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd
from ase import units
from ase.calculators.calculator import Calculator, all_changes

import nff.utils.constants as const
from nff.md.colvars import ColVar

nonbondedMethod = {
    "NonPeriodic": app.CutoffNonPeriodic,
}


class PropertyNotPresent(Exception):
    pass


class BiasBase(Calculator):
    """Basic Calculator class with neural force field

    Args:
        model: the deural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
    """

    implemented_properties = [  # noqa
        "energy",
        "forces",
        "energy_unbiased",
        "forces_unbiased",
        "cv_vals",
        "ext_pos",
        "cv_invmass",
        "grad_length",
        "cv_grad_lengths",
        "cv_dot_PES",
        "const_vals",
    ]

    def __init__(
        self,
        mmparms,
        cv_defs: List[Dict],
        equil_temp: float = 300.0,
        extra_constraints: Optional[List[Dict]] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        # OpenMM setup
        if "prmtop" in mmparms:
            parm = pmd.load_file(mmparms["prmtop"])
        elif "parm7" in mmparms:
            # for a list with parm7 and rst7
            parm = pmd.load_file(mmparms["parm7"], mmparms["rst7"])
        else:
            raise NotImplementedError
        # in case we need PBC, the pdb contains the box values
        app.PDBFile(mmparms["pdb"])

        system = parm.createSystem(
            nonbondedMethod=nonbondedMethod[mmparms["nonbonded"]],
            nonbondedCutoff=mmparms["nonbonded_cutoff"] * unit.nanometers,
        )
        platform = omm.Platform.getPlatformByName(mmparms["platform"])
        # this will not be used
        integrator = omm.LangevinIntegrator(0.0 * unit.kelvin, 1.0 / unit.picoseconds, 0.1 * unit.picoseconds)
        self.context = omm.Context(system, integrator, platform)

        # BiasBase setup
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(ColVar(cv_def["definition"]))

        self.equil_temp = equil_temp

        self.ext_coords = np.zeros(shape=(self.num_cv, 1))
        self.ext_masses = np.zeros(shape=(self.num_cv, 1))
        self.ext_forces = np.zeros(shape=(self.num_cv, 1))
        self.ext_vel = np.zeros(shape=(self.num_cv, 1))
        self.ext_binwidth = np.zeros(shape=(self.num_cv, 1))
        self.ext_k = np.zeros(shape=(self.num_cv,))
        self.ext_dt = 0.0

        self.ranges = np.zeros(shape=(self.num_cv, 2))
        self.margins = np.zeros(shape=(self.num_cv, 1))
        self.conf_k = np.zeros(shape=(self.num_cv, 1))

        for ii, cv in enumerate(self.cv_defs):
            if "range" in cv:
                self.ext_coords[ii] = cv["range"][0]
                self.ranges[ii] = cv["range"]
            else:
                raise PropertyNotPresent("range")

            if "margin" in cv:
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv:
                self.conf_k[ii] = cv["conf_k"]

            if "ext_k" in cv:
                self.ext_k[ii] = cv["ext_k"]
            elif "ext_sigma" in cv:
                self.ext_k[ii] = (units.kB * self.equil_temp) / (cv["ext_sigma"] * cv["ext_sigma"])
            else:
                raise PropertyNotPresent("ext_k/ext_sigma")

            self.cv_defs[ii]["type"] = cv.get("type", "not_angle")

        self.constraints = None
        self.num_const = 0
        if extra_constraints is not None:
            self.constraints = []
            for cv in extra_constraints:
                self.constraints.append({})

                self.constraints[-1]["func"] = CV(cv["definition"])

                self.constraints[-1]["pos"] = cv["pos"]
                if "k" in cv:
                    self.constraints[-1]["k"] = cv["k"]
                elif "sigma" in cv:
                    self.constraints[-1]["k"] = (units.kB * self.equil_temp) / (cv["sigma"] * cv["sigma"])
                else:
                    raise PropertyNotPresent("k/sigma")

                self.constraints[-1]["type"] = cv.get("type", "not_angle")

            self.num_const = len(self.constraints)

        self.cvs = np.zeros(shape=(self.num_cv, 1))
        self.cv_grads = np.zeros(shape=(self.num_cv, atoms.get_positions().shape[0], atoms.get_positions().shape[1]))
        self.cv_grad_lens = np.zeros(shape=(self.num_cv, 1))
        self.cv_invmass = np.zeros(shape=(self.num_cv, 1))
        self.cv_dot_PES = np.zeros(shape=(self.num_cv, 1))

    def _update_bias(self, xi: np.ndarray):
        pass

    def _propagate_ext(self):
        pass

    def _up_extvel(self):
        pass

    def _check_boundaries(self, xi: np.ndarray):
        in_bounds = (xi <= self.ranges[:, 1]).all() and (xi >= self.ranges[:, 0]).all()
        return in_bounds

    def diff(self, a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str) -> Union[np.ndarray, float]:
        """get difference of elements of numbers or arrays
        in range(-inf, inf) if is_angle is False or in range(-pi, pi) if is_angle is True
        Args:
            a: number or array
            b: number or array
        Returns:
            diff: element-wise difference (a-b)
        """
        diff = a - b

        # wrap to range(-pi,pi) for angle
        if isinstance(diff, np.ndarray) and cv_type == "angle":
            diff[diff > np.pi] -= 2 * np.pi
            diff[diff < -np.pi] += 2 * np.pi

        elif cv_type == "angle":
            if diff < -np.pi:
                diff += 2 * np.pi
            elif diff > np.pi:
                diff -= 2 * np.pi

        return diff

    def step_bias(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of bias

        Args:
            curr_cv: current value of the cv
            cv_index: for multidimensional FES

        Returns:
            bias_ener: bias energy
            bias_grad: gradiant of the bias in CV space, needs to be dotted with the cv_gradient
        """

        self._propagate_ext()

        bias_grad = np.zeros_like(grad_xi[0])
        bias_ener = 0.0

        for i in range(self.num_cv):
            # harmonic coupling of extended coordinate to reaction coordinate
            dxi = self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_grad += self.ext_k[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ext_k[i] * dxi**2

            # harmonic walls for confinement to range of interest
            if self.ext_coords[i] > (self.ranges[i][1] + self.margins[i]):
                r = self.diff(self.ranges[i][1] + self.margins[i], self.ext_coords[i], self.cv_defs[i]["type"])
                self.ext_forces[i] += self.conf_k[i] * r

            elif self.ext_coords[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(self.ranges[i][0] - self.margins[i], self.ext_coords[i], self.cv_defs[i]["type"])
                self.ext_forces[i] += self.conf_k[i] * r

        self._update_bias(xi)
        self._up_extvel()

        return bias_ener, bias_grad

    def harmonic_constraint(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """energy and gradient of additional harmonic constraint

        Args:
            xi: current value of constraint "CV"
            grad_xi: Cartesian gradient of these CVs

        Returns:
            constr_ener: constraint energy
            constr_grad: gradient of the constraint energy

        """

        constr_grad = np.zeros_like(grad_xi[0])
        constr_ener = 0.0

        for i in range(self.num_const):
            dxi = self.diff(xi[i], self.constraints[i]["pos"], self.constraints[i]["type"])
            constr_grad += self.constraints[i]["k"] * dxi * grad_xi[i]
            constr_ener += 0.5 * self.constraints[i]["k"] * dxi**2

        return constr_ener, constr_grad

    def calculate(
        self,
        atoms=None,
        properties=[
            "energy",
            "forces",
            "energy_unbiased",
            "forces_unbiased",
            "cv_vals",
            "cv_invmass",
            "grad_length",
            "cv_grad_lengths",
            "cv_dot_PES",
            "const_vals",
        ],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties: list of keywords that can be present in self.results
        system_changes (default from ase)
        """

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run OpenMM
        numpy_pos = 0.1 * atoms.get_positions()  # A to nm
        new_positions = ([omm.Vec3(*xyz.tolist()) for xyz in numpy_pos]) * unit.nanometers
        self.context.setPositions(new_positions)
        state = self.context.getState(getForces=True, getEnergy=True)
        model_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories / unit.moles) / const.EV_TO_KCAL_MOL
        model_forces = (
            state.getForces(asNumpy=True).value_in_unit(unit.kilocalories / (unit.moles * unit.angstroms))
            / const.EV_TO_KCAL_MOL
        )

        inv_masses = 1.0 / atoms.get_masses()
        M_inv = np.diag(np.repeat(inv_masses, 3).flatten())

        for ii, _ in enumerate(self.cv_defs):
            xi, xi_grad = self.the_cv[ii](atoms)
            self.cvs[ii] = xi
            self.cv_grads[ii] = xi_grad
            self.cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            self.cv_invmass[ii] = np.matmul(xi_grad.flatten(), np.matmul(M_inv, xi_grad.flatten()))
            self.cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_forces.flatten())

        bias_ener, bias_grad = self.step_bias(self.cvs, self.cv_grads)
        energy = model_energy + bias_ener
        forces = model_forces - bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(self.num_const, atoms.get_positions().shape[0], atoms.get_positions().shape[1])
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)

            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            forces -= const_grad

        self.results = {
            "energy": energy.reshape(-1),
            "forces": forces.reshape(-1, 3),
            "energy_unbiased": model_energy,
            "forces_unbiased": model_forces.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_forces),
            "cv_vals": self.cvs,
            "cv_grad_lengths": self.cv_grad_lens,
            "cv_invmass": self.cv_invmass,
            "cv_dot_PES": self.cv_dot_PES,
            "ext_pos": self.ext_coords,
        }

        if self.constraints:
            self.results["const_vals"] = consts


class eABF(BiasBase):
    """extended-system Adaptive Biasing Force Calculator
       class with neural force field

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Lagevin dyn of extended system
        (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
    """

    def __init__(
        self,
        mmparms,
        cv_defs: list[dict],
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        **kwargs,
    ):
        BiasBase.__init__(self, mmparms=mmparms, cv_defs=cv_defs, equil_temp=equil_temp, **kwargs)

        self.ext_dt = dt * units.fs
        self.nfull = nfull

        for ii, cv in enumerate(self.cv_defs):
            if "bin_width" in cv:
                self.ext_binwidth[ii] = cv["bin_width"]
            elif "ext_sigma" in cv:
                self.ext_binwidth[ii] = cv["ext_sigma"]
            else:
                raise PropertyNotPresent("bin_width")

            if "ext_pos" in cv:
                # set initial position
                self.ext_coords[ii] = cv["ext_pos"]
            else:
                raise PropertyNotPresent("ext_pos")

            if "ext_mass" in cv:
                self.ext_masses[ii] = cv["ext_mass"]
            else:
                raise PropertyNotPresent("ext_mass")

        # initialize extended system at target temp of MD simulation
        for i in range(self.num_cv):
            self.ext_vel[i] = np.random.randn() * np.sqrt(self.equil_temp * units.kB / self.ext_masses[i])

        self.friction = friction_per_ps * 1.0e-3 / units.fs
        self.rand_push = np.sqrt(self.equil_temp * self.friction * self.ext_dt * units.kB / (2.0e0 * self.ext_masses))
        self.prefac1 = 2.0 / (2.0 + self.friction * self.ext_dt)
        self.prefac2 = (2.0e0 - self.friction * self.ext_dt) / (2.0e0 + self.friction * self.ext_dt)

        # set up all grid accumulators for ABF
        self.nbins_per_dim = np.array([1 for i in range(self.num_cv)])
        self.grid = []
        for i in range(self.num_cv):
            self.nbins_per_dim[i] = int(np.ceil(np.abs(self.ranges[i, 1] - self.ranges[i, 0]) / self.ext_binwidth[i]))
            self.grid.append(
                np.arange(
                    self.ranges[i, 0] + self.ext_binwidth[i] / 2,
                    self.ranges[i, 1],
                    self.ext_binwidth[i],
                )
            )
        self.nbins = np.prod(self.nbins_per_dim)

        # accumulators and conditional averages
        self.bias = np.zeros((self.num_cv, *self.nbins_per_dim), dtype=float)
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)

        self.cv_crit = np.copy(self.bias)

        self.histogram = np.zeros(self.nbins_per_dim, dtype=float)
        self.ext_hist = np.zeros_like(self.histogram)

    def get_index(self, xi: np.ndarray) -> tuple:
        """get list of bin indices for current position of CVs or extended variables
        Args:
            xi (np.ndarray): Current value of collective variable
        Returns:
            bin_x (list):
        """
        bin_x = np.zeros(shape=xi.shape, dtype=np.int64)
        for i in range(self.num_cv):
            bin_x[i] = int(np.floor(np.abs(xi[i] - self.ranges[i, 0]) / self.ext_binwidth[i]))
        return tuple(bin_x.reshape(1, -1)[0])

    def _update_bias(self, xi: np.ndarray):
        if self._check_boundaries(self.ext_coords):
            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1

            # linear ramp function
            ramp = 1.0 if self.ext_hist[bink] > self.nfull else self.ext_hist[bink] / self.nfull

            for i in range(self.num_cv):
                # apply bias force on extended system
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i] * self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"]),
                )
                self.ext_forces[i] -= ramp * self.bias[i][bink]

        """
        Not sure how this can be dumped/printed to work with the rest
        # xi-conditioned accumulators for CZAR
        if (xi <= self.ranges[:,1]).all() and
               (xi >= self.ranges[:,0]).all():

            bink = self.get_index(xi)
            self.histogram[bink] += 1

            for i in range(self.num_cv):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]],
                          self.cv_defs[i]['type'])
                self.correction_czar[i][bink] += self.ext_k[i] * dx
        """

    def _propagate_ext(self):
        self.ext_rand_gauss = np.random.randn(len(self.ext_vel), 1)

        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses
        self.ext_coords += self.prefac1 * self.ext_dt * self.ext_vel

        # wrap to range(-pi,pi) for angle
        for ii in range(self.num_cv):
            if self.cv_defs[ii]["type"] == "angle":
                if self.ext_coords[ii] > np.pi:
                    self.ext_coords[ii] -= 2 * np.pi
                elif self.ext_coords[ii] < -np.pi:
                    self.ext_coords[ii] += 2 * np.pi

    def _up_extvel(self):
        self.ext_vel *= self.prefac2
        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses


class WTMeABF(eABF):
    """Well tempered MetaD extended-system Adaptive Biasing Force Calculator
       based on eABF class

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Langevin dyn of extended system
        (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
        hill_height: unscaled height of the MetaD Gaussian hills in eV
        hill_drop_freq: #steps between depositing Gaussians
        well_tempered_temp: ficticious temperature for the well-tempered scaling
    """

    def __init__(
        self,
        mmparms,
        cv_defs: list[dict],
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        hill_height: float = 0.0,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 4000.0,
        **kwargs,
    ):
        eABF.__init__(
            self,
            mmparms=mmparms,
            cv_defs=cv_defs,
            equil_temp=equil_temp,
            dt=dt,
            friction_per_ps=friction_per_ps,
            nfull=nfull,
            **kwargs,
        )

        self.hill_height = hill_height
        self.hill_drop_freq = hill_drop_freq
        self.hill_std = np.zeros(self.num_cv)
        self.hill_var = np.zeros(self.num_cv)
        self.well_tempered_temp = well_tempered_temp
        self.call_count = 0
        self.center = []

        for ii, cv in enumerate(self.cv_defs):
            if "hill_std" in cv:
                self.hill_std[ii] = cv["hill_std"]
                self.hill_var[ii] = cv["hill_std"] * cv["hill_std"]
            else:
                raise PropertyNotPresent("hill_std")

        # set up all grid for MetaD potential
        self.metapot = np.zeros_like(self.histogram)
        self.metaforce = np.zeros_like(self.bias)

    def _update_bias(self, xi: np.ndarray):
        mtd_forces = self.get_wtm_force(self.ext_coords)
        self.call_count += 1

        if self._check_boundaries(self.ext_coords):
            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1

            # linear ramp function
            ramp = 1.0 if self.ext_hist[bink] > self.nfull else self.ext_hist[bink] / self.nfull

            for i in range(self.num_cv):
                # apply bias force on extended system
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i] * self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"]),
                )
                self.ext_forces[i] -= ramp * self.bias[i][bink] + mtd_forces[i]

    def get_wtm_force(self, xi: np.ndarray) -> np.ndarray:
        """compute well-tempered metadynamics bias force from superposition of gaussian hills
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        is_in_bounds = self._check_boundaries(xi)

        if (self.call_count % self.hill_drop_freq == 0) and is_in_bounds:
            self.center.append(np.copy(xi.reshape(-1)))

        if is_in_bounds and self.num_cv == 1:
            bias_force, _ = self._accumulate_wtm_force(xi)
        else:
            bias_force, _ = self._analytic_wtm_force(xi)

        return bias_force

    def _accumulate_wtm_force(self, xi: np.ndarray) -> Tuple[list, float]:
        """compute numerical WTM bias force from a grid
        Right now this works only for 1D CVs
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        bink = self.get_index(xi)
        if self.call_count % self.hill_drop_freq == 0:
            w = self.hill_height * np.exp(-self.metapot[bink] / (units.kB * self.well_tempered_temp))

            dx = self.diff(self.grid[0], xi[0], self.cv_defs[0]["type"]).reshape(
                -1,
            )
            epot = w * np.exp(-(dx * dx) / (2.0 * self.hill_var[0]))
            self.metapot += epot
            self.metaforce[0] -= epot * dx / self.hill_var[0]

        return self.metaforce[:, bink], self.metapot[bink]

    def _analytic_wtm_force(self, xi: np.ndarray) -> Tuple[list, float]:
        """compute analytic WTM bias force from sum of gaussians hills
        Args:
            xi: state of collective variable
        Returns:
            bias_force: bias force from metadynamics
        """

        local_pot = 0.0
        bias_force = np.zeros(shape=(self.num_cv))

        # this should never be the case!
        if len(self.center) == 0:
            print(" >>> Warning: no metadynamics hills stored")
            return bias_force

        ind = np.ma.indices((len(self.center),))[0]
        ind = np.ma.masked_array(ind)

        dist_to_centers = np.array(
            [self.diff(xi[ii], np.asarray(self.center)[:, ii], self.cv_defs[ii]["type"]) for ii in range(self.num_cv)]
        )

        if self.num_cv > 1:
            ind[(abs(dist_to_centers) > 3 * self.hill_std.reshape(-1, 1)).all(axis=0)] = np.ma.masked
        else:
            ind[(abs(dist_to_centers) > 3 * self.hill_std.reshape(-1, 1)).all(axis=0)] = np.ma.masked

        # can get slow in long run, so only iterate over significant elements
        for i in np.nditer(ind.compressed(), flags=["zerosize_ok"]):
            w = self.hill_height * np.exp(-local_pot / (units.kB * self.well_tempered_temp))

            epot = w * np.exp(-np.power(dist_to_centers[:, i] / self.hill_std, 2).sum() / 2.0)
            local_pot += epot
            bias_force -= epot * dist_to_centers[:, i] / self.hill_var

        return bias_force.reshape(-1, 1), local_pot


def welford_var(count: float, mean: float, M2: float, newValue: float) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm
    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample
    Returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var
