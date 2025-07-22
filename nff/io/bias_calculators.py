from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes

import nff.utils.constants as const
from nff.io.ase_calcs import NeuralFF, check_directed
from nff.md.colvars import ColVar as CV
from nff.nn.models.cp3d import OnlyBondUpdateCP3D
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.schnet_features import SchNetFeatures
from nff.utils.cuda import batch_to

DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0
UNDIRECTED = [SchNet, SchNetDiabat, HybridGraphConv, SchNetFeatures, OnlyBondUpdateCP3D]


class BiasBase(NeuralFF):
    """Basic Calculator class with neural force field

    Args:
        model: the neural force field model
        cv_def: list of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        equil_temp: float temperature of the simulation (important for extended system dynamics)
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
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
        model,
        cv_defs: List[dict],
        equil_temp: float = 300.0,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        extra_constraints: Optional[List[Dict]] = None,
        **kwargs,
    ):
        NeuralFF.__init__(self, model=model, device=device, en_key=en_key, directed=directed, **kwargs)

        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(CV(cv_def["definition"]))

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
                raise KeyError("range")

            if "margin" in cv:
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv:
                self.conf_k[ii] = cv["conf_k"]

            if "ext_k" in cv:
                self.ext_k[ii] = cv["ext_k"]
            elif "ext_sigma" in cv:
                self.ext_k[ii] = (units.kB * self.equil_temp) / (cv["ext_sigma"] * cv["ext_sigma"])
            else:
                raise KeyError("ext_k/ext_sigma")

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
                    raise KeyError("k/sigma")

                self.constraints[-1]["type"] = cv.get("type", "not_angle")

            self.num_const = len(self.constraints)

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
        bias_ener, bias_grad = self._extended_dynamics(xi, grad_xi)

        self._update_bias(xi)
        self._up_extvel()

        return bias_ener, bias_grad

    def _extended_dynamics(
        self,
        xi: np.ndarray,
        grad_xi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                r = self.diff(
                    self.ranges[i][1] + self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r

            elif self.ext_coords[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(
                    self.ranges[i][0] - self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r

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

        if not any(isinstance(self.model, i) for i in UNDIRECTED):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        if "forces" in self.properties:
            kwargs["requires_forces"] = True
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # change energy and force to numpy array and eV
        model_energy = prediction[self.en_key].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)

        gradient = prediction.get(grad_key)
        forces = prediction.get("forces")
        if gradient is not None:
            model_grad = gradient.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        elif forces is not None:
            model_grad = - forces.detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        else:
            raise KeyError(grad_key)

        inv_masses = 1.0 / atoms.get_masses()
        M_inv = np.diag(np.repeat(inv_masses, 3).flatten())

        cvs = np.zeros(shape=(self.num_cv, 1))
        cv_grads = np.zeros(
            shape=(
                self.num_cv,
                atoms.get_positions().shape[0],
                atoms.get_positions().shape[1],
            )
        )
        cv_grad_lens = np.zeros(shape=(self.num_cv, 1))
        cv_invmass = np.zeros(shape=(self.num_cv, 1))
        cv_dot_PES = np.zeros(shape=(self.num_cv, 1))
        for ii, _ in enumerate(self.cv_defs):
            xi, xi_grad = self.the_cv[ii](atoms)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.einsum("i,ii,i", xi_grad.flatten(), M_inv, xi_grad.flatten())
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        self.results = {
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
        }

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(
                    self.num_const,
                    atoms.get_positions().shape[0],
                    atoms.get_positions().shape[1],
                )
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)

            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            grad += const_grad

        self.results.update(
            {
                "energy": energy.reshape(-1),
                "forces": -grad.reshape(-1, 3),
                "ext_pos": self.ext_coords,
            }
        )

        if self.constraints:
            self.results["const_vals"] = consts

        if requires_stress:
            stress = prediction["stress_volume"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
            self.results["stress"] = stress * (1 / atoms.get_volume())


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
        model,
        cv_defs: List[dict],
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        BiasBase.__init__(
            self,
            cv_defs=cv_defs,
            equil_temp=equil_temp,
            model=model,
            device=device,
            en_key=en_key,
            directed=directed,
            **kwargs,
        )

        self.ext_dt = dt * units.fs
        self.nfull = nfull

        for ii, cv in enumerate(self.cv_defs):
            if "bin_width" in cv:
                self.ext_binwidth[ii] = cv["bin_width"]
            elif "ext_sigma" in cv:
                self.ext_binwidth[ii] = cv["ext_sigma"]
            else:
                raise KeyError("bin_width")

            if "ext_pos" in cv:
                # set initial position
                self.ext_coords[ii] = cv["ext_pos"]
            else:
                raise KeyError("ext_pos")

            if "ext_mass" in cv:
                self.ext_masses[ii] = cv["ext_mass"]
            else:
                raise KeyError("ext_mass")

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


class aMDeABF(eABF):
    """Accelerated extended-system Adaptive Biasing Force Calculator
    class with neural force field

    Accelerated Molecular Dynamics

    see:
        aMD: Hamelberg et. al., J. Chem. Phys. 120, 11919 (2004); https://doi.org/10.1063/1.1755656
        GaMD: Miao et. al., J. Chem. Theory Comput. (2015); https://doi.org/10.1021/acs.jctc.5b00436
        SaMD: Zhao et. al., J. Phys. Chem. Lett. 14, 4, 1103 - 1112 (2023); https://doi.org/10.1021/acs.jpclett.2c03688

    Apply global boost potential to potential energy, that is independent of Collective Variables.

    Args:
        model: the neural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        amd_parameter: acceleration parameter; SaMD, GaMD == sigma0; aMD == alpha
        init_step: initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps: equilibration steps, min, max and var of potential energy is still updated
                          force constant of coupling is calculated from previous steps
        amd_method: "aMD": apply accelerated MD
                    "GaMD_lower": use lower bound for GaMD boost
                    "GaMD_upper: use upper bound for GaMD boost
                    "SaMD: apply Sigmoid accelerated MD
        equil_temp: float temperature of the simulation (important for extended system dynamics)
        dt: time step of the extended dynamics (has to be equal to that of the real system dyn!)
        friction_per_ps: friction for the Lagevin dyn of extended system
        (has to be equal to that of the real system dyn!)
        nfull: numer of samples need for full application of bias force
    """

    def __init__(
        self,
        model,
        cv_defs: List[dict],
        dt: float,
        friction_per_ps: float,
        amd_parameter: float,
        collect_pot_samples: bool,
        estimate_k: bool,
        apply_amd: bool,
        amd_method: str = "gamd_lower",
        samd_c0: float = 0.0001,
        equil_temp: float = 300.0,
        nfull: int = 100,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        super().__init__(
            model=model,
            cv_defs=cv_defs,
            dt=dt,
            friction_per_ps=friction_per_ps,
            equil_temp=equil_temp,
            nfull=nfull,
            device=device,
            en_key=en_key,
            directed=directed,
            **kwargs,
        )

        self.amd_parameter = amd_parameter
        self.collect_pot_samples = collect_pot_samples
        self.estimate_k = estimate_k
        self.apply_amd = apply_amd

        self.amd_method = amd_method.lower()

        assert self.amd_method in [
            "gamd_upper",
            "gamd_lower",
            "samd",
            "amd",
        ], f"Unknown aMD method {self.amd_method}"

        if self.amd_method == "amd":
            print(" >>> Warning: Please use GaMD or SaMD to obtain accurate free energy estimates!\n")

        self.pot_count = 0
        self.pot_var = 0.0
        self.pot_std = 0.0
        self.pot_m2 = 0.0
        self.pot_avg = 0.0
        self.pot_min = +np.inf
        self.pot_max = -np.inf
        self.k0 = 0.0
        self.k1 = 0.0
        self.k = 0.0
        self.E = 0.0
        self.c0 = samd_c0
        self.c = 1 / self.c0 - 1

        self.amd_pot = 0.0
        self.amd_pot_traj = []

        self.amd_c1 = np.zeros_like(self.histogram)
        self.amd_c2 = np.zeros_like(self.histogram)
        self.amd_m2 = np.zeros_like(self.histogram)
        self.amd_corr = np.zeros_like(self.histogram)

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

        epot = self.results["energy_unbiased"]
        self.amd_forces = np.copy(self.results["forces_unbiased"])

        self._propagate_ext()
        bias_ener, bias_grad = self._extended_dynamics(xi, grad_xi)

        if self.collect_pot_samples is True:
            self._update_pot_distribution(epot)

        if self.estimate_k is True:
            self._calc_E_k0()

        self._update_bias(xi)

        if self.apply_amd is True:
            # apply amd boost potential only if U0 below bound
            if epot < self.E:
                boost_ener, boost_grad = self._apply_boost(epot)
            else:
                boost_ener, boost_grad = 0.0, 0.0 * self.amd_forces
            bias_ener += boost_ener
            bias_grad += boost_grad

            bink = self.get_index(xi)
            (
                self.amd_c1[bink],
                self.amd_m2[bink],
                self.amd_c2[bink],
            ) = welford_var(
                self.histogram[bink],
                self.amd_c1[bink],
                self.amd_m2[bink],
                boost_ener,
            )

        self._up_extvel()

        return bias_ener, bias_grad

    def _apply_boost(self, epot):
        """Apply boost potential to forces"""
        if self.amd_method == "amd":
            amd_pot = np.square(self.E - epot) / (self.parameter + (self.E - epot))
            boost_grad = (
                ((epot - self.E) * (epot - 2.0 * self.parameter - self.E)) / np.square(epot - self.parameter - self.E)
            ) * self.amd_forces

        elif self.amd_method == "samd":
            amd_pot = self.amd_pot = (
                self.pot_max
                - epot
                - 1
                / self.k
                * np.log(
                    (self.c + np.exp(self.k * (self.pot_max - self.pot_min)))
                    / (self.c + np.exp(self.k * (epot - self.pot_min)))
                )
            )
            boost_grad = (
                -(1.0 / (np.exp(-self.k * (epot - self.pot_min) + np.log((1 / self.c0) - 1)) + 1) - 1) * self.amd_forces
            )

        else:
            prefac = self.k0 / (self.pot_max - self.pot_min)
            amd_pot = 0.5 * prefac * np.power(self.E - epot, 2)
            boost_grad = prefac * (self.E - epot) * self.amd_forces

        return amd_pot, boost_grad

    def _update_pot_distribution(self, epot: float):
        """update min, max, avg, var and std of epot

        Args:
            epot: potential energy
        """
        self.pot_min = np.min([epot, self.pot_min])
        self.pot_max = np.max([epot, self.pot_max])
        self.pot_count += 1
        self.pot_avg, self.pot_m2, self.pot_var = welford_var(self.pot_count, self.pot_avg, self.pot_m2, epot)
        self.pot_std = np.sqrt(self.pot_var)

    def _calc_E_k0(self):
        """compute force constant for amd boost potential

        Args:
            epot: potential energy
        """
        if self.amd_method == "gamd_lower":
            self.E = self.pot_max
            ko = (self.amd_parameter / self.pot_std) * ((self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg))

            self.k0 = np.min([1.0, ko])

        elif self.amd_method == "gamd_upper":
            ko = (1.0 - self.amd_parameter / self.pot_std) * (
                (self.pot_max - self.pot_min) / (self.pot_avg - self.pot_min)
            )
            if 0.0 < ko <= 1.0:
                self.k0 = ko
            else:
                self.k0 = 1.0
            self.E = self.pot_min + (self.pot_max - self.pot_min) / self.k0

        elif self.amd_method == "samd":
            ko = (self.amd_parameter / self.pot_std) * ((self.pot_max - self.pot_min) / (self.pot_max - self.pot_avg))

            self.k0 = np.min([1.0, ko])
            if (self.pot_std / self.amd_parameter) <= 1.0:
                self.k = self.k0
            else:
                self.k1 = np.max(
                    [
                        0,
                        (np.log(self.c) + np.log((self.pot_std) / (self.amd_parameter) - 1))
                        / (self.pot_avg - self.pot_min),
                    ]
                )
                self.k = np.max([self.k0, self.k1])

        elif self.amd_method == "amd":
            self.E = self.pot_max

        else:
            raise ValueError(f" >>> Error: unknown aMD method {self.amd_method}!")


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
        model,
        cv_defs: List[dict],
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        hill_height: float = 0.0,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 4000.0,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        eABF.__init__(
            self,
            cv_defs=cv_defs,
            equil_temp=equil_temp,
            dt=dt,
            friction_per_ps=friction_per_ps,
            nfull=nfull,
            model=model,
            device=device,
            en_key=en_key,
            directed=directed,
            **kwargs,
        )

        self.hill_height = hill_height
        self.hill_drop_freq = hill_drop_freq
        self.hill_std = np.zeros(shape=(self.num_cv))
        self.hill_var = np.zeros(shape=(self.num_cv))
        self.well_tempered_temp = well_tempered_temp
        self.call_count = 0
        self.center = []

        for ii, cv in enumerate(self.cv_defs):
            if "hill_std" in cv:
                self.hill_std[ii] = cv["hill_std"]
                self.hill_var[ii] = cv["hill_std"] * cv["hill_std"]
            else:
                raise KeyError("hill_std")

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


class AttractiveBias(NeuralFF):
    """Biased Calculator that introduces an attractive term
       Designed to be used with UQ as CV

    Args:
        model: the neural force field model
        cv_def: list of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        gamma: coupling strength, regulates strength of attraction
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
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
        model,
        cv_defs: List[dict],
        gamma=1.0,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        extra_constraints: Optional[List[Dict]] = None,
        **kwargs,
    ):
        NeuralFF.__init__(self, model=model, device=device, en_key=en_key, directed=directed, **kwargs)

        self.gamma = gamma
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(CV(cv_def["definition"]))

        self.ext_coords = np.zeros(shape=(self.num_cv, 1))
        self.ranges = np.zeros(shape=(self.num_cv, 2))
        self.margins = np.zeros(shape=(self.num_cv, 1))
        self.conf_k = np.zeros(shape=(self.num_cv, 1))

        for ii, cv in enumerate(self.cv_defs):
            if "range" in cv:
                self.ext_coords[ii] = cv["range"][0]
                self.ranges[ii] = cv["range"]
            else:
                raise KeyError("range")

            if "margin" in cv:
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv:
                self.conf_k[ii] = cv["conf_k"]

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
                    raise KeyError("k/sigma")

                self.constraints[-1]["type"] = cv.get("type", "not_angle")

            self.num_const = len(self.constraints)

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

        bias_grad = -(self.gamma * grad_xi).sum(axis=0)
        bias_ener = -(self.gamma * xi).sum()

        # harmonic walls for confinement to range of interest
        for i in range(self.num_cv):
            if xi[i] > (self.ranges[i][1] + self.margins[i]):
                r = self.diff(
                    self.ranges[i][1] + self.margins[i],
                    xi[i],
                    self.cv_defs[i]["type"],
                )
                bias_grad += self.conf_k[i] * r
                bias_ener += 0.5 * self.conf_k[i] * r**2

            elif xi[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(
                    self.ranges[i][0] - self.margins[i],
                    xi[i],
                    self.cv_defs[i]["type"],
                )
                bias_grad += self.conf_k[i] * r
                bias_ener += 0.5 * self.conf_k[i] * r**2

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

        if not any(isinstance(self.model, i) for i in UNDIRECTED):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # change energy and force to numpy array and eV
        model_energy = prediction[self.en_key].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)

        if grad_key in prediction:
            model_grad = prediction[grad_key].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
        else:
            raise KeyError(grad_key)

        inv_masses = 1.0 / atoms.get_masses()
        M_inv = np.diag(np.repeat(inv_masses, 3).flatten())

        cvs = np.zeros(shape=(self.num_cv, 1))
        cv_grads = np.zeros(
            shape=(
                self.num_cv,
                atoms.get_positions().shape[0],
                atoms.get_positions().shape[1],
            )
        )
        cv_grad_lens = np.zeros(shape=(self.num_cv, 1))
        cv_invmass = np.zeros(shape=(self.num_cv, 1))
        cv_dot_PES = np.zeros(shape=(self.num_cv, 1))
        for ii, _ in enumerate(self.cv_defs):
            xi, xi_grad = self.the_cv[ii](atoms)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.einsum("i,ii,i", xi_grad.flatten(), M_inv, xi_grad.flatten())
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        self.results = {
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
        }

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(
                    self.num_const,
                    atoms.get_positions().shape[0],
                    atoms.get_positions().shape[1],
                )
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)

            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            grad += const_grad

        self.results.update(
            {
                "energy": energy.reshape(-1),
                "forces": -grad.reshape(-1, 3),
                "ext_pos": self.ext_coords,
            }
        )

        if self.constraints:
            self.results["const_vals"] = consts

        if requires_stress:
            stress = prediction["stress_volume"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
            self.results["stress"] = stress * (1 / atoms.get_volume())


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
