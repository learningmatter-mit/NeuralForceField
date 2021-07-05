import numpy as np
import pickle
import os
import copy
import json
import random
import math
import argparse
from functools import partial

from ase.io.trajectory import Trajectory
from ase import Atoms

from nff.train import load_model
from nff.utils import constants as const
from nff.md.utils_ax import atoms_to_nxyz
from nff.md.tully.io import get_results, load_json, get_atoms
from nff.md.tully.step import (
    try_hop,
    verlet_step_1, verlet_step_2,
    truhlar_decoherence,
    # add_decoherence
    diabatic_c, compute_T
)
from nff.md.nvt_ax import NoseHoover, NoseHooverChain

# TO-DO:
# - Figure out if the three-step propagator is actually working
# - Fix p_hop for multiple states
# - Check everything in detail
# - Add decoherence

METHOD_DIC = {
    "nosehoover": NoseHoover,
    "nosehooverchain": NoseHooverChain
}

DECOHERENCE_DIC = {"truhlar": truhlar_decoherence}

TULLY_LOG_FILE = 'tully.log'
TULLY_SAVE_FILE = 'tully.pickle'
TULLY_MINIMAL_SAVE = 'tully_small.pickle'


class NeuralTully:
    def __init__(self,
                 atoms_list,
                 device,
                 batch_size,
                 num_states,
                 initial_surf,
                 dt,
                 elec_substeps,
                 max_time,
                 cutoff,
                 model_path,
                 diabat_keys,
                 explicit_diabat_prop,
                 log_all_hops=False,
                 cutoff_skin=1.0,
                 max_gap_hop=0.018,
                 nbr_update_period=20,
                 save_period=30,
                 decoherence=None,
                 **kwargs):
        """
        `max_gap_hop` in a.u.
        """

        self.atoms_list = atoms_list
        self.vel = self.get_vel()
        self.T = None
        self.model = self.load_model(model_path)

        self.t = 0
        self.props = None
        self.num_atoms = len(self.atoms_list[0])
        self.num_samples = len(atoms_list)
        self.diabat_keys = diabat_keys
        self.num_diabat = len(diabat_keys)

        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.batch_size = batch_size
        self.num_states = num_states
        self.surfs = np.ones(self.num_samples,
                             dtype=np.int) * initial_surf

        self.dt = dt * const.FS_TO_AU
        self.elec_substeps = elec_substeps

        self.max_time = max_time * const.FS_TO_AU
        self.nbr_update_period = nbr_update_period
        self.nbr_list = None
        self.max_gap_hop = max_gap_hop

        self.save_file = TULLY_SAVE_FILE
        self.minimal_save_file = TULLY_MINIMAL_SAVE

        self.log_file = TULLY_LOG_FILE
        self.save_period = save_period

        self.log_template = self.setup_logging()
        self.p_hop = 0
        self.any_hopped = False
        self.explicit_diabat = explicit_diabat_prop
        self.log_all_hops = log_all_hops
        self.c = self.init_c()

        self.clear_pickles()

        self.decoherence = self.init_decoherence(params=decoherence)

        # if not self.decoherence:
        #     return

        # self.delta_R = 0
        # self.delta_P = 0
        # self.sigma = self.init_sigma()

    def init_decoherence(self, params):
        if not params:
            return

        name = params['name']
        kwargs = params.get('kwargs', {})

        method = DECOHERENCE_DIC[name]
        func = partial(method,
                       **kwargs)

        return func

    def clear_pickles(self):
        for file in [self.save_file, self.minimal_save_file,
                     self.log_file]:
            if os.path.isfile(file):
                os.remove(file)

    def load_model(self, model_path):
        param_path = os.path.join(model_path, 'params.json')
        with open(param_path, 'r') as f:
            params = json.load(f)

        model = load_model(model_path, params, params['model_type'])

        return model

    def get_vel(self):
        vel = np.stack([atoms.get_velocities()
                        for atoms in self.atoms_list])
        vel /= const.BOHR_RADIUS * const.ASE_TO_FS * const.FS_TO_AU

        return vel

    def init_c(self):
        if self.explicit_diabat:
            num_states = self.num_diabat
        else:
            num_states = self.num_states

        c = np.zeros((self.num_samples,
                      num_states),
                     dtype='complex128')
        c[:, self.surfs[0]] = 1
        return c

    def init_sigma(self):
        """
        Electronic density matrix
        """

        # c has dimension num_samples x num_states
        # sigma has dimension num_samples x num_states
        # x num_states

        sigma = np.ones((self.num_samples,
                         self.num_states,
                         self.num_states),
                        dtype=np.complex128)

        sigma = self.c.reshape(self.num_samples,
                               self.num_states,
                               1)
        sigma *= np.conj(self.c.reshape(self.num_samples,
                                        1,
                                        self.num_states))

        return sigma

    @property
    def U(self):
        if not self.props:
            return
        return self.props["U"]

    @property
    def forces(self):
        _forces = np.stack([-self.props[f'energy_{i}_grad']
                            for i in range(self.num_states)],
                           axis=1)
        _forces = (_forces.reshape(self.num_samples, -1,
                                   self.num_states, 3)
                   .transpose(0, 2, 1, 3))

        return _forces

    @property
    def energy(self):
        _energy = np.stack([self.props[f'energy_{i}'].reshape(-1)
                            for i in range(self.num_states)],
                           axis=1)
        return _energy

    @property
    def full_energy(self):
        _energy = np.stack([self.props[f'energy_{i}'].reshape(-1)
                            for i in range(self.num_diabat)],
                           axis=1)
        return _energy

    @property
    def nacv(self):
        _nacv = np.zeros((self.num_samples, self.num_states,
                          self.num_states, self.num_atoms, 3))
        for i in range(self.num_states):
            for j in range(self.num_states):
                if i == j:
                    continue
                _nacv[:, i, j, :] = self.props[f'nacv_{i}{j}']
        return _nacv

    @property
    def gap(self):
        num_samples = self.energy.shape[0]
        num_states = self.energy.shape[1]

        _gap = np.zeros((num_samples, num_states, num_states))
        _gap -= self.energy.reshape(num_samples, 1, num_states)
        _gap += self.energy.reshape(num_samples, num_states, 1)

        return _gap

    @property
    def force_nacv(self):

        # self.gap has shape num_samples x num_states x num_states
        # `nacv` has shape num_samples x num_states x num_states
        # x num_atoms x 3

        gap = self.gap.reshape(self.num_samples,
                               self.num_states,
                               self.num_states,
                               1,
                               1)

        _force_nacv = self.nacv * gap

        return _force_nacv

    @property
    def mass(self):
        _mass = (self.atoms_list[0].get_masses()
                 * const.AMU_TO_AU)
        return _mass

    @property
    def nxyz(self):
        _nxyz = np.stack([atoms_to_nxyz(atoms) for atoms in
                          self.atoms_list])

        return _nxyz

    @property
    def xyz(self):
        _xyz = self.nxyz[..., 1:]
        return _xyz

    @xyz.setter
    def xyz(self, val):
        for atoms, xyz in zip(self.atoms_list, val):
            atoms.set_positions(xyz)

    @property
    def pot_V(self):
        """
        Potential energy matrix
        """

        V = np.zeros((self.num_samples,
                      self.num_states,
                      self.num_states))
        idx = np.arange(self.num_states)
        np.put_along_axis(
            V,
            idx.reshape(1, -1, 1),
            self.energy.reshape(self.num_samples,
                                self.num_states,
                                1),
            axis=2
        )

        return V

    @property
    def full_pot_V(self):
        """
        Potential energy matrix
        """

        V = np.zeros((self.num_samples,
                      self.num_diabat,
                      self.num_diabat))
        idx = np.arange(self.num_diabat)
        np.put_along_axis(
            V,
            idx.reshape(1, -1, 1),
            self.full_energy.reshape(self.num_samples,
                                     self.num_diabat,
                                     1),
            axis=2
        )

        return V

    @property
    def H_d(self):
        _H_d = np.zeros((self.num_samples,
                         self.num_diabat,
                         self.num_diabat))

        for i in range(self.num_diabat):
            for j in range(self.num_diabat):
                diabat_key = self.diabat_keys[i][j]
                _H_d[..., i, j] = (self.props[diabat_key]
                                   .reshape(-1))

        return _H_d

    @property
    def unique_diabats(self):
        keys = np.array(self.diabat_keys).reshape(-1).tolist()
        keys = list(set(keys))
        return keys

    @property
    def state_dict(self):
        _state_dict = {"nxyz": self.nxyz,
                       "nacv": self.nacv,
                       "force_nacv": self.force_nacv,
                       "energy": self.energy,
                       "forces": self.forces,
                       "U": self.U,
                       "t": self.t / const.FS_TO_AU,
                       "vel": self.vel,
                       "c": self.c,
                       "T": self.T,
                       "surfs": self.surfs}
        return _state_dict

    @property
    def minimal_state_dict(self):
        keys = ['nxyz', 'energy', 'surfs', 't']
        minimal = {key: self.state_dict[key]
                   for key in keys}

        return minimal

    def save(self):
        with open(self.save_file, "ab") as f:
            pickle.dump(self.state_dict, f)

        with open(self.minimal_save_file, "ab") as f:
            pickle.dump(self.minimal_state_dict, f)

    def setup_logging(self):

        states = [f"State {i}" for i in range(self.num_states)]
        hdr = "%-9s " % "Time [fs]"
        for state in states:
            hdr += "%15s " % state
        hdr += "%15s " % "|c|"
        hdr += "%15s " % "Hop prob."

        with open(self.log_file, 'w') as f:
            f.write(hdr)

        template = "%-10.1f "
        for i, state in enumerate(states):
            template += "%15.4f%%"
        template += "%15.4f"
        template += "%15.4f"

        return template

    def log(self):
        time = self.t / const.FS_TO_AU
        pcts = []
        for i in range(self.num_states):
            num_surf = (self.surfs == i).sum()
            pct = num_surf / self.num_samples * 100
            pcts.append(pct)

        norm_c = np.mean(np.linalg.norm(self.c, axis=1))
        p_avg = np.mean(np.max(self.p_hop, axis=1))
        text = self.log_template % (time, *pcts,
                                    norm_c, p_avg)

        with open(self.log_file, 'a') as f:
            f.write("\n" + text)

        # print(self.c[0])

    @classmethod
    def from_pickles(cls,
                     file,
                     max_time=None):

        state_dicts = []
        with open(file, 'rb') as f:
            while True:
                try:
                    state_dict = pickle.load(f)
                    state_dicts.append(state_dict)
                except EOFError:
                    break
                time = state_dict['t']
                if max_time is not None and time > max_time:
                    break

        num_samples = state_dicts[0]['nxyz'].shape[0]
        trjs = [[] for _ in range(num_samples)]

        for state_dict in state_dicts:
            for i, nxyz in enumerate(state_dict['nxyz']):
                atoms = Atoms(nxyz[:, 0],
                              positions=nxyz[:, 1:])
                trjs[i].append(atoms)

        return state_dicts, trjs

    def update_props(self,
                     needs_nbrs):

        props = get_results(model=self.model,
                            nxyz=self.nxyz,
                            nbr_list=self.nbr_list,
                            num_atoms=self.num_atoms,
                            needs_nbrs=needs_nbrs,
                            cutoff=self.cutoff,
                            cutoff_skin=self.cutoff_skin,
                            device=self.device,
                            batch_size=self.batch_size,
                            old_U=self.U,
                            num_states=self.num_states,
                            surf=self.surfs,
                            max_gap_hop=self.max_gap_hop,
                            # all_grads=self.decoherence
                            all_grads=True,
                            diabat_keys=self.unique_diabats
                            )

        self.props = props

    def step(self, needs_nbrs):

        # inner loop to propagate electronic wavefunction
        # for _ in range(self.elec_substeps):
        # c, T = runge_c(c=self.c,
        #                vel=self.vel,
        #                nacv=self.nacv,
        #                energy=self.energy,
        #                elec_dt=self.dt / self.elec_substeps,
        #                hbar=1)

        # c, delta_P, delta_R = add_decoherence(c=c,
        #                                       surfs=self.surfs,
        #                                       new_surfs=new_surfs,
        #                                       delta_P=self.delta_P,
        #                                       delta_R=self.delta_R,
        #                                       nacv=self.nacv,
        #                                       energy=self.energy,
        #                                       forces=self.forces,
        #                                       mass=self.mass)

        # self.delta_P = delta_P
        # self.delta_R = delta_R

        # if any sample hopped then it was within the cutoff
        # gap, which means we have forces for the other states
        # which we can use for the rest of this inner loop

        # self.c = c
        # self.T = T
        # self.t += self.dt / self.elec_substeps

        old_H_d = copy.deepcopy(self.H_d)
        old_H_ad = copy.deepcopy(self.pot_V)
        old_U = copy.deepcopy(self.U)

        # xyz converted to a.u. for the step and then
        # back to Angstrom after
        new_xyz, new_vel = verlet_step_1(self.forces,
                                         self.surfs,
                                         vel=self.vel,
                                         xyz=self.xyz / const.BOHR_RADIUS,
                                         mass=self.mass,
                                         dt=self.dt)
        self.xyz = new_xyz * const.BOHR_RADIUS
        self.vel = new_vel

        self.update_props(needs_nbrs)
        new_vel = verlet_step_2(forces=self.forces,
                                surfs=self.surfs,
                                vel=self.vel,
                                xyz=self.xyz,
                                mass=self.mass,
                                dt=self.dt)
        self.vel = new_vel

        # propagate c in diabatic basis
        self.c = diabatic_c(c=self.c,
                            old_H_d=old_H_d,
                            new_H_d=self.H_d,
                            old_H_ad=old_H_ad,
                            new_H_ad=self.pot_V,
                            old_U=old_U,
                            new_U=self.U,
                            dt=self.dt,
                            elec_substeps=self.elec_substeps,
                            explicit_diabat=self.explicit_diabat)

        self.T, _ = compute_T(nacv=self.nacv,
                              vel=self.vel,
                              c=self.c)

        c = self.c[:, :self.num_states]
        new_surfs, new_vel, p_hop = try_hop(c=c,
                                            T=self.T,
                                            dt=self.dt,
                                            surfs=self.surfs,
                                            vel=self.vel,
                                            nacv=self.nacv,
                                            mass=self.mass,
                                            energy=self.energy,
                                            max_gap_hop=self.max_gap_hop)

        self.any_hopped = (new_surfs != self.surfs).any()
        self.surfs = new_surfs
        self.vel = new_vel
        self.p_hop = p_hop
        self.t += self.dt

        self.log()

        if not self.decoherence:
            return

        self.c = self.decoherence(c=self.c,
                                  surfs=self.surfs,
                                  energy=self.energy,
                                  vel=self.vel,
                                  dt=self.dt,
                                  mass=self.mass)

    def run(self):
        steps = math.ceil(self.max_time / self.dt)
        epochs = math.ceil(steps / self.nbr_update_period)

        counter = 0

        self.model.to(self.device)
        self.update_props(needs_nbrs=True)

        for _ in range(epochs):
            for i in range(self.nbr_update_period):
                needs_nbrs = (i == 0)
                self.step(needs_nbrs=needs_nbrs)

                if counter % self.save_period == 0:
                    self.save()

                # save the geoms if any just hopped
                elif self.any_hopped and self.log_all_hops:
                    self.save()

                counter += 1

        with open(self.log_file, 'a') as f:
            f.write('\nNeural Tully terminated normally.')


class CombinedNeuralTully:
    def __init__(self,
                 atoms,
                 ground_params,
                 tully_params):

        self.ground_dynamics = self.init_ground(atoms=atoms,
                                                ground_params=ground_params)
        self.ground_params = ground_params
        self.ground_savefile = ground_params.get("savefile")

        self.tully_params = tully_params
        self.num_trj = tully_params['num_trj']

    def init_ground(self,
                    atoms,
                    ground_params):

        ase_ground_params = copy.deepcopy(ground_params)
        ase_ground_params["trajectory"] = ground_params.get("savefile")
        logfile = ase_ground_params['logfile']

        if os.path.isfile(logfile):
            os.remove(logfile)

        method = METHOD_DIC[ase_ground_params["thermostat"]]
        ground_dynamics = method(atoms, **ase_ground_params)

        return ground_dynamics

    def sample_ground_geoms(self):
        steps = math.ceil(self.ground_params["max_time"] /
                          self.ground_params["timestep"])
        equil_steps = math.ceil(self.ground_params["equil_time"] /
                                self.ground_params["timestep"])

        self.ground_dynamics.run(steps=steps)

        trj = Trajectory(self.ground_savefile)

        loginterval = self.ground_params.get("loginterval", 1)
        logged_equil = math.ceil(equil_steps / loginterval)
        possible_states = [trj[index] for index in
                           range(logged_equil, len(trj))]
        random_indices = random.sample(range(len(possible_states)),
                                       self.num_trj)
        actual_states = [possible_states[index] for index in random_indices]

        return actual_states

    def run(self):
        atoms_list = self.sample_ground_geoms()
        tully = NeuralTully(atoms_list=atoms_list,
                            **self.tully_params)
        tully.run()

    @classmethod
    def from_file(cls,
                  file):

        all_params = load_json(file)
        atomsbatch = get_atoms(all_params)
        ground_params = all_params['ground_params']

        tully_params = all_params['tully_params']
        model_path = os.path.join(all_params['weightpath'],
                                  str(all_params["nnid"]))
        tully_params.update({"model_path": model_path,
                             "device": all_params["device"],
                             "diabat_keys": all_params["diabat_keys"]})

        instance = cls(atoms=atomsbatch,
                       ground_params=ground_params,
                       tully_params=tully_params)

        return instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file',
                        type=str,
                        help='Info file with parameters',
                        default='job_info.json')
    args = parser.parse_args()

    path = args.params_file
    combined_tully = CombinedNeuralTully.from_file(path)

    try:
        combined_tully.run()
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()


if __name__ == '__main__':
    main()
