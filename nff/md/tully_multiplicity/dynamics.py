"""
Script for running Tully surface hopping dynamics. Note
that PyTorch version >=1.9 is required for the matrix
exponentiation in the propagator. Older versions were not
build for complex numbers, and so their matrix exponentials
of complex numbers return nonsense. Numpy does not have
matrix exponentiation and SciPy can only do one batch at a
time, so we need to use PyTorch to do it efficiently.
"""

import copy
import json
import math
import os
import pickle
import random
import shutil
from functools import partial
from typing import *

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from nff.md.nvt_ax import NoseHoover, NoseHooverChain
from nff.md.tully_multiplicity.io import get_atoms, get_results, load_json
from nff.md.tully_multiplicity.step import (
    adiabatic_c,
    get_p_hop,
    truhlar_decoherence,
    try_hop,
    verlet_step_1,
    verlet_step_2,
)
from nff.md.utils_ax import atoms_to_nxyz
from nff.train import load_model
from nff.utils import constants as const

METHOD_DIC = {"nosehoover": NoseHoover, "nosehooverchain": NoseHooverChain}

DECOHERENCE_DIC = {"truhlar": truhlar_decoherence}

TULLY_LOG_FILE = "tully.log"
TULLY_SAVE_FILE = "tully.pickle"

MODEL_KWARGS = {"add_nacv": False, "add_grad": True, "inference": True, "en_keys_for_grad": ["energy_0"]}


class NeuralTully:
    def __init__(
        self,
        atoms_list,
        device,
        batch_size,
        adiabatic_keys: List[str],
        initial_surf: str,
        dt,
        elec_substeps: int,
        max_time,
        cutoff,
        model_paths: List,
        simple_vel_scale,
        hop_eqn,
        cutoff_skin,
        max_gap_hop,
        nbr_update_period,
        save_period,
        decoherence,
        **kwargs,
    ):
        """
        `max_gap_hop` in a.u.
        """

        self.atoms_list = atoms_list
        self.vel = self.get_vel()

        self.device = device
        self.models = [self.load_model(model_path).to(device) for model_path in model_paths]

        self.T = None
        self.U_old = None

        self.t = 0
        self.props = {}
        self.num_atoms = len(self.atoms_list[0])
        self.num_samples = len(atoms_list)
        self.adiabatic_keys = adiabatic_keys
        self.repeated_keys = []
        self.spinadiabatic_to_adiabatic = {}
        self.spinadiabatic_to_statenum = {}
        self.spinadiabatic_keys = []
        for key in adiabatic_keys:
            if "S" in key:
                new_key = key + "_ms+0"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
            elif "D" in key:
                new_key = key + "_ms-1/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms+1/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
            elif "T" in key:
                new_key = key + "_ms-1"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms+0"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms+1"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
            elif "Q" in key:
                new_key = key + "_ms-3/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms-1/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms+1/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key
                new_key = key + "_ms+3/2"
                self.repeated_keys.append(key)
                self.spinadiabatic_to_statenum[new_key] = len(self.spinadiabatic_keys)
                self.spinadiabatic_keys.append(new_key)
                self.spinadiabatic_to_adiabatic[new_key] = key

        self.num_spinadibat = len(self.spinadiabatic_keys)
        self.num_states = len(self.spinadiabatic_keys)
        self.initial_surf_num = self.spinadiabatic_to_statenum[initial_surf]
        self.statenum_to_spinadiabatic = {v: k for k, v in self.spinadiabatic_to_statenum.items()}
        # The dictionary "statenum_to_spinadiabatic" will be important to
        # understand the output of the simulation
        json_object = json.dumps(self.statenum_to_spinadiabatic, indent=4)
        with open("statenum_to_spinadiabatic.json", "w") as outf:
            outf.write(json_object)

        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.batch_size = batch_size

        self.dt = dt * const.FS_TO_AU
        self.elec_substeps = elec_substeps

        self.max_time = max_time * const.FS_TO_AU
        self.nbr_update_period = nbr_update_period
        self.nbr_list = None
        self.max_gap_hop = max_gap_hop

        self.log_file = TULLY_LOG_FILE
        self.save_file = TULLY_SAVE_FILE
        self.save_period = save_period

        self.log_template = self.setup_logging()
        self.p_hop = np.zeros((self.num_samples, self.num_states))
        self.just_hopped = None
        self.surfs = np.ones(self.num_samples, dtype=np.int) * self.initial_surf_num
        self.c_hmc = self.init_c()

        self.update_props(needs_nbrs=True)
        self.c_diag = self.get_c_diag()

        # sanity check
        if not (self.surfs == np.argmax(np.abs(self.c_diag), axis=1)).all():
            print("The states in the diagonal basis got reordered! Adjusting surfs!")
            self.surfs = np.argmax(np.abs(self.c_diag), axis=1)
            print(self.surfs)

        self.setup_save()
        self.decoherence = self.init_decoherence(params=decoherence)
        self.decoherence_type = decoherence["name"]
        self.hop_eqn = hop_eqn
        self.simple_vel_scale = simple_vel_scale

        if os.path.isfile(TULLY_SAVE_FILE):
            self.restart()

    def setup_save(self):
        if os.path.isfile(self.save_file):
            os.remove(self.save_file)

    def init_decoherence(self, params):
        if not params:
            return None

        name = params["name"]
        kwargs = params.get("kwargs", {})

        method = DECOHERENCE_DIC[name]
        func = partial(method, **kwargs)

        return func

    def load_model(self, model_path):
        param_path = os.path.join(model_path, "params.json")
        with open(param_path, "r") as f:
            params = json.load(f)

        model = load_model(model_path, params, params["model_type"])

        return model

    @property
    def mass(self):
        _mass = self.atoms_list[0].get_masses() * const.AMU_TO_AU

        return _mass

    @property
    def nxyz(self):
        _nxyz = np.stack([atoms_to_nxyz(atoms) for atoms in self.atoms_list])

        return _nxyz

    @nxyz.setter
    def nxyz(self, _nxyz):
        for atoms, this_nxyz in zip(self.atoms_list, _nxyz):
            atoms.set_positions(this_nxyz[:, 1:])

    @property
    def xyz(self):
        _xyz = self.nxyz[..., 1:]

        return _xyz

    @xyz.setter
    def xyz(self, val):
        for atoms, xyz in zip(self.atoms_list, val):
            atoms.set_positions(xyz)

    def get_vel(self):
        vel = np.stack([atoms.get_velocities() for atoms in self.atoms_list])
        vel /= const.BOHR_RADIUS * const.ASE_TO_FS * const.FS_TO_AU

        return vel

    def init_c(self):
        c = np.zeros((self.num_samples, self.num_states), dtype="complex128")
        c[:, self.surfs[0]] = 1
        return c

    def get_c_hmc(self):
        """
        state coefficients in the HMC basis
        """
        c_hmc = np.einsum("ijk,ik->ij", self.U, self.c_diag)

        return c_hmc

    def get_c_diag(self):
        """
        state coefficients in the diagonal basis
        """
        c_diag = np.einsum("ijk,ik->ij", self.U.conj().transpose(0, 2, 1), self.c_hmc)

        return c_diag

    def get_forces(self):
        _forces = np.stack([-self.props[f"energy_{key}_grad"] for key in self.repeated_keys], axis=1)
        _forces = _forces.reshape(self.num_samples, -1, self.num_states, 3).transpose(0, 2, 1, 3)

        return _forces

    def get_energy(self):
        _energy = np.stack([self.props[f"energy_{key}"].reshape(-1) for key in self.repeated_keys], axis=1)

        return _energy

    def get_nacv(self):
        _nacv = np.zeros((self.num_samples, self.num_states, self.num_states, self.num_atoms, 3))
        for state_n1 in range(self.num_states):
            state1 = self.statenum_to_spinadiabatic[state_n1]
            splits = state1.split("_")
            adiabat1 = splits[0]
            splits[1]

            for state_n2 in range(self.num_states):
                state2 = self.statenum_to_spinadiabatic[state_n2]
                splits = state2.split("_")
                adiabat2 = splits[0]
                splits[1]

                if adiabat1 == adiabat2:
                    continue
                if adiabat1[0] != adiabat2[0]:
                    # checks for the same degeneracy
                    continue

                key = f"NACV_{adiabat1}_to_{adiabat2}_grad"
                if key not in self.props:
                    continue
                _nacv[:, state_n1, state_n2, :] = self.props[key]
                _nacv[:, state_n2, state_n1, :] = -self.props[key]

        return _nacv

    def get_gap(self):
        num_samples = self.energy.shape[0]
        num_states = self.energy.shape[1]

        _gap = np.zeros((num_samples, num_states, num_states))
        _gap -= self.energy.reshape(num_samples, 1, num_states)
        _gap += self.energy.reshape(num_samples, num_states, 1)

        return _gap

    def get_force_nacv(self):
        # self.gap has shape num_samples x num_states x num_states
        # `nacv` has shape num_samples x num_states x num_states
        # x num_atoms x 3

        nacv = self.nacv
        if nacv is None:
            return None

        gap = self.gap.reshape(self.num_samples, self.num_states, self.num_states, 1, 1)

        _force_nacv = -nacv * gap

        return _force_nacv

    def get_pot_V(self):
        """
        Potential energy matrix in n_adiabat x n_adiabat space
        """

        V = np.zeros((self.num_samples, self.num_states, self.num_states))
        idx = np.arange(self.num_states)
        np.put_along_axis(V, idx.reshape(1, -1, 1), self.energy.reshape(self.num_samples, self.num_states, 1), axis=2)

        return V

    def get_SOC_mat(self):
        """
        Matrix with SOCs in HMC basis
        """

        H_soc = np.zeros((self.num_samples, self.num_states, self.num_states), dtype=np.complex128)

        for state_n1 in range(self.num_states):
            state1 = self.statenum_to_spinadiabatic[state_n1]
            splits = state1.split("_")
            adiabat1 = splits[0]
            spin_ms1 = splits[1]

            for state_n2 in range(self.num_states):
                state2 = self.statenum_to_spinadiabatic[state_n2]
                splits = state2.split("_")
                adiabat2 = splits[0]
                spin_ms2 = splits[1]

                try:
                    a = self.props[f"SOC_{adiabat1}_to_{adiabat2}_a"]
                    b = self.props[f"SOC_{adiabat1}_to_{adiabat2}_b"]
                    c = self.props[f"SOC_{adiabat1}_to_{adiabat2}_c"]
                except BaseException:
                    continue

                ST_soc = False
                TT_soc = False
                if "S" in adiabat1:
                    ST_soc = True
                elif "T" in adiabat1 and "T" in adiabat2:
                    TT_soc = True
                else:
                    raise NotImplementedError

                if ST_soc:
                    if spin_ms2 == "ms-1":
                        soc_val = a + 1j * b
                    elif spin_ms2 == "ms+0":
                        soc_val = 0.0 + 1j * c
                    elif spin_ms2 == "ms+1":
                        soc_val = a - 1j * b

                elif TT_soc:
                    if spin_ms1 == "ms-1" and spin_ms2 == "ms-1":
                        soc_val = 0.0 + 1j * c
                    elif spin_ms1 == "ms-1" and spin_ms2 == "ms+0":
                        soc_val = -a + 1j * b
                    elif spin_ms1 == "ms+0" and spin_ms2 == "ms-1":
                        soc_val = a + 1j * b
                    elif spin_ms1 == "ms+0" and spin_ms2 == "ms+1":
                        soc_val = -a + 1j * b
                    elif spin_ms1 == "ms+1" and spin_ms2 == "ms+0":
                        soc_val = a + 1j * b
                    elif spin_ms1 == "ms+1" and spin_ms2 == "ms+1":
                        soc_val = 0.0 - 1j * c

                H_soc[:, state_n1, state_n2] = soc_val
                H_soc[:, state_n2, state_n1] = soc_val.conj()

        return H_soc

    def get_H_hmc(self):
        """
        Sum of potential energy matrix and SOCs
        """

        V = self.pot_V.astype(np.complex128)
        H_hmc = V + self.SOC_mat

        return H_hmc

    def get_U(self):
        """
        Diagonalizes H^total
        """

        eVals, U = np.linalg.eigh(self.H_hmc)

        return U, eVals

    def get_H_plus_nacv(self):
        if self.nacv is None:
            return None
        # pot_V = self.pot_V
        H_hmc = self.H_hmc
        nac_term = -1j * (self.nacv * self.vel.reshape(self.num_samples, 1, 1, self.num_atoms, 3)).sum((-1, -2))

        # return pot_V + nac_term
        return H_hmc + nac_term

    def get_neg_G_hmc(self):
        neg_G = np.zeros((self.num_samples, self.num_states, self.num_states, self.num_atoms, 3))
        idx = np.arange(self.num_states)
        np.put_along_axis(
            neg_G,
            idx.reshape(1, -1, 1, 1, 1),
            self.forces.reshape(self.num_samples, self.num_states, 1, self.num_atoms, 3),
            axis=2,
        )
        neg_G += self.force_nacv

        return neg_G

    def get_neg_G_diag(self):
        neg_G_diag = np.einsum("ijk,ikl...,ilm->ijm...", self.U.conj().transpose((0, 2, 1)), self.neg_G_hmc, self.U)

        return neg_G_diag

    def get_diag_energy(self):
        H_diag = np.einsum("ijk,ikl,ilm->ijm", self.U.conj().transpose((0, 2, 1)), self.H_hmc, self.U)
        idxs = np.arange(self.num_states)
        _energy = np.take_along_axis(np.real(H_diag), idxs.reshape(1, -1, 1), axis=2)

        return _energy.reshape(self.num_samples, self.num_states)

    def get_diag_forces(self):
        diag_forces = np.diagonal(self.neg_G_diag, axis1=1, axis2=2).transpose((0, 3, 1, 2))

        return np.real(diag_forces)

    @property
    def state_dict(self):
        _state_dict = {
            "nxyz": self.nxyz,
            "nacv": self.nacv,
            # "force_nacv": self.force_nacv,
            "energy": self.energy,
            "forces": self.forces,
            # "H_d": self.H_d,
            "U": self.U,
            "t": self.t / const.FS_TO_AU,
            "vel": self.vel,
            "c_hmc": self.c_hmc,
            "c_diag": self.c_diag,
            # "T": self.T,
            "surfs": self.surfs,
        }
        return _state_dict

    @state_dict.setter
    def state_dict(self, dic):
        for key, val in dic.items():
            if key in ["force_nacv"]:
                continue
            setattr(self, key, val)
        self.t *= const.FS_TO_AU

    def save(self, idx=None):
        if idx is None:
            with open(self.save_file, "ab") as f:
                pickle.dump(self.state_dict, f)
            return

        if idx.size == 0:
            return

        state_dict = self.state_dict
        use_dict = {}
        idx = set(idx)

        for key, val in state_dict.items():
            if key == "t":
                continue
            if val is None:
                continue

            use_val = []
            for i, v in enumerate(val):
                this_val = v if (i in idx) else None
                use_val.append(this_val)
            use_dict[key] = use_val
        use_dict["t"] = state_dict["t"]

        with open(self.save_file, "ab") as f:
            pickle.dump(use_dict, f)

    def restart(self):
        state_dicts, _ = NeuralTully.from_pickle(TULLY_SAVE_FILE)
        self.state_dict = state_dicts[-1]

    def setup_logging(self, remove_old=True):
        states = [f"State {i}" for i in range(self.num_states)]
        hdr = "%-9s " % "Time [fs]"
        for state in states:
            hdr += "%15s " % state
        hdr += "%15s " % "|c|"
        hdr += "%15s " % "Hop prob."

        if not os.path.isfile(self.log_file) or remove_old:
            with open(self.log_file, "w") as f:
                f.write(hdr)

        template = "%-10.2f "
        for i, state in enumerate(states):
            template += "%15.6f"
        template += "%15.4f"
        template += "%15.4f"

        return template

    def clean_c_p(self):
        c_states = self.c.shape[-1]
        c = self.c[np.bitwise_not(np.isnan(self.c))].reshape(-1, c_states)

        p_states = self.p_hop.shape[-1]
        p_nan_idx = np.isnan(self.p_hop).any(-1)
        p_fine_idx = np.bitwise_not(p_nan_idx)
        p = self.p_hop[p_fine_idx].reshape(-1, p_states)

        return c, p

    def log(self):
        time = self.t / const.FS_TO_AU
        #         pcts = []
        #         for i in range(self.num_states):
        #             num_surf = (self.surfs == i).sum()
        #             pct = num_surf / self.num_samples * 100
        #             pcts.append(pct)

        pcts_hmc = []
        argmax = np.argmax(np.abs(self.c_hmc), axis=1)
        for i in range(self.num_states):
            num_surf = (argmax == i).sum()
            pct = num_surf / self.num_samples  # * 100
            pcts_hmc.append(pct)

        # c, p = self.clean_c_p()
        norm_c = np.mean(np.linalg.norm(self.c_hmc, axis=1))
        p_avg = np.mean(np.max(self.p_hop, axis=1))
        text = self.log_template % (
            time,  # *pcts,
            *pcts_hmc,
            norm_c,
            p_avg,
        )

        with open(self.log_file, "a") as f:
            f.write("\n" + text)

        # sanity check
        if norm_c > 2.0:
            print("Norm of coefficients too large!!")
            exit(1)

    @classmethod
    def from_pickle(cls, file, max_time=None):
        state_dicts = []
        with open(file, "rb") as f:
            while True:
                try:
                    state_dict = pickle.load(f)
                    state_dicts.append(state_dict)
                except EOFError:
                    break
                time = state_dict["t"]
                if max_time is not None and time > max_time:
                    break

        sample_nxyz = state_dicts[0]["nxyz"]
        # whether this is a single trajectory or multiple
        single = len(sample_nxyz.shape) == 2
        num_samples = 1 if single else sample_nxyz.shape[0]
        trjs = [[] for _ in range(num_samples)]

        for state_dict in state_dicts:
            nxyz = state_dict["nxyz"]
            if single:
                nxyz = [nxyz]
            for i, nxyz in enumerate(nxyz):
                if nxyz is None:
                    trjs[i].append(None)
                    continue
                atoms = Atoms(nxyz[:, 0], positions=nxyz[:, 1:])
                trjs[i].append(atoms)

        if single:
            trjs = trjs[0]

        return state_dicts, trjs

    def update_props(self, needs_nbrs):
        props = get_results(
            models=self.models,
            nxyz=self.nxyz,
            nbr_list=self.nbr_list,
            num_atoms=self.num_atoms,
            needs_nbrs=needs_nbrs,
            cutoff=self.cutoff,
            cutoff_skin=self.cutoff_skin,
            device=self.device,
            batch_size=self.batch_size,
        )

        self.props = props
        self.update_selfs()

    def update_selfs(self):
        # simple reorganiation of NN outputs
        self.energy = self.get_energy()
        self.forces = self.get_forces()
        self.nacv = self.get_nacv()
        self.gap = self.get_gap()
        self.force_nacv = self.get_force_nacv()
        self.pot_V = self.get_pot_V()

        # assembly of complicated matrices
        self.SOC_mat = self.get_SOC_mat()
        self.H_hmc = self.get_H_hmc()
        self.H_plus_nacv = self.get_H_plus_nacv()

        # diagonalization of HMC representation
        self.U, evals = self.get_U()

        if self.U_old is not None:
            # the following is an implementation of Appendix B
            # from Mai, Marquetand, Gonzalez
            # Int.J. Quant. Chem. 2015, 115, 1215-1231
            V = np.einsum("ijk,ikl->ijl", self.U.conj().transpose((0, 2, 1)), self.U_old)

            # attempt to make V diagonally dominant
            for replica in range(self.num_samples):
                abs_v = np.abs(V[replica])
                arg_max = np.argmax(abs_v, axis=1)
                # sanity check print statement
                # if len(np.unique(arg_max)) < len(arg_max):
                #     print("V could not be made diagonal dominant!")

                for column in range(self.num_states):
                    curr_col = copy.deepcopy(V[replica][:, column])
                    new_col = copy.deepcopy(V[replica][:, arg_max[column]])
                    # switch columns
                    V[replica][:, column] = new_col
                    V[replica][:, arg_max[column]] = curr_col

            # (CV)_{ab} = V_{ab} delta(Hdiag_aa - Hdiag_bb)
            # setting everything to zero where
            # the difference in diagonal elements is NOT zero
            # for replica, hdiag in enumerate(evals):
            hdiag = evals.reshape((self.num_samples, self.num_states, 1))
            diff = hdiag - hdiag.transpose((0, 2, 1))
            preserved_idxs = np.isclose(diff, np.zeros(shape=diff.shape), atol=1e-8, rtol=0.0)
            V[~preserved_idxs] = 0.0

            # Loewding symmetric orthonormalization
            u, s, vh = np.linalg.svd(V)
            Phi_adj = np.einsum("ijk, ikl->ijl", u, vh)

            corrected_U = np.einsum("ijk, ikl->ijl", self.U, Phi_adj)
            self.U = copy.deepcopy(corrected_U)

            # check eq B11
            epsilon = 0.1  # hardcoded for now
            diagonals = np.einsum("ijj->ij", np.einsum("ijk,ikl->ijl", self.U.conj().transpose((0, 2, 1)), self.U_old))
            ((1 - epsilon) < diagonals).all()
            # if not anti_hermitian:
            #     print("WARNING: Time step likely too large! At least one new unitary matrix ",
            #           "does not fulfill anti-hermicity!")
            #     print(f"epsilon = {epsilon}")
            #     print("diagonal elements:\n", diagonals)
            #     print("H_diag:\n", evals)
            #     print(V)
            #     print(preserved_idxs)

        self.U_old = copy.deepcopy(self.U)

        self.neg_G_hmc = self.get_neg_G_hmc()
        self.neg_G_diag = self.get_neg_G_diag()
        self.diag_energy = self.get_diag_energy()
        self.diag_forces = self.get_diag_forces()

    def do_hop(self, old_c, new_c, P):
        self.p_hop = get_p_hop(hop_eqn=self.hop_eqn, old_c=old_c, new_c=new_c, P=P, surfs=self.surfs)

        new_surfs, new_vel = try_hop(
            p_hop=self.p_hop,
            surfs=self.surfs,
            vel=self.vel,
            nacv=self.nacv,
            mass=self.mass,
            energy=self.energy,
            max_gap_hop=self.max_gap_hop,
            simple_scale=self.simple_vel_scale,
        )

        return new_surfs, new_vel

    def add_decoherence(self):
        if not self.decoherence:
            return

        self.c_diag = self.decoherence(
            c=self.c_diag, surfs=self.surfs, energy=self.diag_energy, vel=self.vel, dt=self.dt, mass=self.mass
        )

        self.c_hmc = self.get_c_hmc()

    def step(self, needs_nbrs):
        copy.deepcopy(self.pot_V)
        copy.deepcopy(self.H_hmc)
        old_H_plus_nacv = copy.deepcopy(self.H_plus_nacv)
        old_U = copy.deepcopy(self.U)

        copy.deepcopy(self.c_hmc)
        old_c_diag = copy.deepcopy(self.c_diag)

        # xyz converted to a.u. for the step and then
        # back to Angstrom after
        new_xyz, new_vel = verlet_step_1(
            self.diag_forces, self.surfs, vel=self.vel, xyz=self.xyz / const.BOHR_RADIUS, mass=self.mass, dt=self.dt
        )
        self.xyz = new_xyz * const.BOHR_RADIUS
        self.vel = new_vel

        # from here on everything is "new"
        self.update_props(needs_nbrs)

        new_vel = verlet_step_2(forces=self.diag_forces, surfs=self.surfs, vel=self.vel, mass=self.mass, dt=self.dt)
        self.vel = new_vel

        self.c_hmc, self.P_hmc = adiabatic_c(
            c=self.c_hmc,
            elec_substeps=self.elec_substeps,
            old_H_plus_nacv=old_H_plus_nacv,
            new_H_plus_nacv=self.H_plus_nacv,
            dt=self.dt,
        )

        #         print("Norm before/after elec substeps (hop):\n",
        #               np.linalg.norm(old_c_hmc, axis=1),
        #               np.linalg.norm(self.c_hmc, axis=1))

        self.c_diag = self.get_c_diag()
        self.P_diag = np.einsum("ijk,ikl,ilm->ijm", self.U.conj().transpose((0, 2, 1)), self.P_hmc, old_U)

        #         if self.nacv is not None:
        #             self.T, _ = compute_T(nacv=self.nacv,
        #                                   vel=self.vel,
        #                                   c=self.c_hmc)

        new_surfs, new_vel = self.do_hop(old_c=old_c_diag, new_c=self.c_diag, P=self.P_diag)

        self.just_hopped = (new_surfs != self.surfs).nonzero()[0]
        # if self.just_hopped.any():

        self.surfs = new_surfs
        self.vel = new_vel
        self.t += self.dt

        self.add_decoherence()
        self.log()

    def run(self):
        steps = math.ceil((self.max_time - self.t) / self.dt)
        epochs = math.ceil(steps / self.nbr_update_period)

        self.save()
        self.log()

        counter = 0

        #         self.model.to(self.device)
        #         if self.t == 0:
        #             self.update_props(needs_nbrs=True)

        for _ in tqdm(range(epochs)):
            for i in range(self.nbr_update_period):
                needs_nbrs = i == 0
                self.step(needs_nbrs=needs_nbrs)
                counter += 1

                if counter % self.save_period == 0:
                    self.save()
        #                 else:
        #                     # save any geoms that just hopped
        #                     self.save(idx=self.just_hopped)

        with open(self.log_file, "a") as f:
            f.write("\nNeural Tully terminated normally.")


class CombinedNeuralTully:
    def __init__(self, atoms, ground_params, tully_params):
        self.reload_ground = tully_params.get("reload_ground", False)
        self.ground_dynamics = self.init_ground(atoms=atoms, ground_params=ground_params)
        self.ground_params = ground_params
        self.ground_savefile = ground_params.get("savefile")

        self.tully_params = tully_params
        self.num_trj = tully_params["num_trj"]

    def init_ground(self, atoms, ground_params):
        ase_ground_params = copy.deepcopy(ground_params)
        ase_ground_params["trajectory"] = ground_params.get("savefile")
        logfile = ase_ground_params["logfile"]
        trj_file = ase_ground_params["trajectory"]

        if os.path.isfile(logfile):
            if self.reload_ground:
                shutil.move(logfile, logfile.replace(".log", "_old.log"))
            else:
                os.remove(logfile)
        if os.path.isfile(trj_file):
            if self.reload_ground:
                shutil.move(trj_file, trj_file.replace(".trj", "_old.trj"))
            else:
                os.remove(trj_file)

        method = METHOD_DIC[ase_ground_params["thermostat"]]
        ground_dynamics = method(atoms, **ase_ground_params)

        return ground_dynamics

    def sample_ground_geoms(self):
        steps = math.ceil(self.ground_params["max_time"] / self.ground_params["timestep"])
        equil_steps = math.ceil(self.ground_params["equil_time"] / self.ground_params["timestep"])
        loginterval = self.ground_params.get("loginterval", 1)

        if self.ground_dynamics is not None:
            old_trj_file = str(self.ground_savefile).replace(".trj", "_old.trj")
            if self.reload_ground and os.path.isfile(old_trj_file):
                trj = Trajectory(old_trj_file)
                atoms = next(iter(reversed(trj)))

                steps -= len(trj) * loginterval
                # set positions and velocities. Don't overwrite atoms because
                # then you lose their calculator
                self.ground_dynamics.atoms.set_positions(atoms.get_positions())
                self.ground_dynamics.atoms.set_velocities(atoms.get_velocities())

            self.ground_dynamics.run(steps=steps)

        trj = Trajectory(self.ground_savefile)

        logged_equil = math.ceil(equil_steps / loginterval)
        possible_states = [trj[index] for index in range(logged_equil, len(trj))]
        random_indices = random.sample(range(len(possible_states)), self.num_trj)
        actual_states = [possible_states[index] for index in random_indices]

        return actual_states

    def run(self):
        atoms_list = self.sample_ground_geoms()
        tully = NeuralTully(atoms_list=atoms_list, **self.tully_params)
        tully.run()

    @classmethod
    def from_file(cls, file):
        all_params = load_json(file)
        ground_params = all_params["ground_params"]
        atomsbatch = get_atoms(all_params=all_params, ground_params=ground_params)
        atomsbatch.calc.model_kwargs = MODEL_KWARGS

        tully_params = all_params["tully_params"]
        if "weightpath" in all_params:
            model_path = os.path.join(all_params["weightpath"], str(all_params["nnid"]))
        else:
            model_path = all_params["model_path"]

        tully_params.update({"model_path": model_path})

        instance = cls(atoms=atomsbatch, ground_params=ground_params, tully_params=tully_params)

        return instance
