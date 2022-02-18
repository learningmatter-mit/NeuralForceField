"""
Ab initio version of Tully's surface hopping
"""

import argparse
import shutil
import os
import math
import numpy as np
from ase import Atoms
import copy

from nff.md.tully.dynamics import (NeuralTully,
                                   TULLY_LOG_FILE,
                                   TULLY_SAVE_FILE)
from nff.md.tully.io import load_json, coords_to_xyz
from nff.md.tully.ab_io import get_results as ab_results
from nff.utils import constants as const


def load_params(file):
    all_params = load_json(file)

    all_params['nacv_details'] = {**all_params,
                                  **all_params['nacv_details']}
    all_params['grad_details'] = {**all_params,
                                  **all_params['grad_details']}

    return all_params


def make_atoms(all_params):
    vel = np.array(all_params['velocities'])
    nxyz = coords_to_xyz(all_params["coords"])
    atoms = Atoms(nxyz[:, 0],
                  positions=nxyz[:, 1:])

    atoms.set_velocities(vel)
    atoms_list = [atoms]

    return atoms_list


class AbTully(NeuralTully):
    def __init__(self,
                 charge,
                 grad_config,
                 nacv_config,
                 grad_details,
                 nacv_details,
                 atoms_list,
                 num_states,
                 initial_surf,
                 dt,
                 max_time,
                 elec_substeps,
                 decoherence,
                 hop_eqn,
                 **kwargs):

        self.atoms_list = atoms_list
        self.vel = self.get_vel()
        self.T = None

        self.t = 0
        self.props = {}
        self.num_atoms = len(self.atoms_list[0])
        self.num_samples = len(atoms_list)
        self.num_states = num_states
        self.surfs = np.ones(self.num_samples,
                             dtype=np.int) * initial_surf

        self.dt = dt * const.FS_TO_AU
        self.elec_substeps = elec_substeps

        self.max_time = max_time * const.FS_TO_AU
        self.max_gap_hop = float('inf')

        self.log_file = TULLY_LOG_FILE
        self.save_file = TULLY_SAVE_FILE

        self.log_template = self.setup_logging(remove_old=False)
        self.p_hop = 0
        self.just_hopped = None
        self.explicit_diabat = False
        self.c = self.init_c()

        self.decoherence = self.init_decoherence(params=decoherence)
        self.decoherence_type = decoherence['name']
        self.hop_eqn = hop_eqn
        self.diabat_propagate = False
        self.simple_vel_scale = False

        self.charge = charge
        self.num_samples = 1

        self.grad_config = grad_config
        self.nacv_config = nacv_config
        self.grad_details = grad_details
        self.nacv_details = nacv_details

        self.step_num = 0

        # only works if you don't use `self.setup_save()`,
        # which deletes the pickle file
        if os.path.isfile(TULLY_SAVE_FILE):
            self.restart()

    @property
    def forces(self):
        inf = np.ones((self.num_atoms,
                       3)) * float('inf')
        _forces = np.stack([-self.props.get(f'energy_{i}_grad',
                                            inf).reshape(-1, 3)
                            for i in range(self.num_states)])
        _forces = _forces.reshape(1, *_forces.shape)

        return _forces

    @forces.setter
    def forces(self, _forces):
        for i in range(self.num_states):
            self.props[f'energy_{i}_grad'] = -_forces[:, i]

    def correct_phase(self,
                      old_force_nacv):

        if old_force_nacv is None:
            return

        new_force_nacv = self.force_nacv
        new_nacv = self.nacv

        delta = np.max(np.linalg.norm(old_force_nacv - new_force_nacv,
                                      axis=((-1, -2))), axis=-1)
        sigma = np.max(np.linalg.norm(old_force_nacv + new_force_nacv,
                                      axis=((-1, -2))), axis=-1)

        delta = delta.reshape(*delta.shape, 1, 1, 1)
        sigma = sigma.reshape(*sigma.shape, 1, 1, 1)

        phase = (-1) ** (delta > sigma)

        print(np.linalg.norm(old_force_nacv - new_force_nacv))
        print(np.linalg.norm(old_force_nacv + new_force_nacv))
        print(phase.squeeze((-1, -2, -3)))

        new_force_nacv = new_force_nacv * phase
        new_nacv = new_nacv * phase

        num_states = new_nacv.shape[1]
        for i in range(num_states):
            for j in range(num_states):
                self.props[f'force_nacv_{i}{j}'] = new_force_nacv[:, i, j]
                self.props[f'nacv_{i}{j}'] = new_nacv[:, i, j]

    def update_props(self,
                     *args,
                     **kwargs):

        old_force_nacv = copy.deepcopy(self.force_nacv)

        job_dir = os.path.join(os.getcwd(), str(self.step_num))
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir)
        else:
            os.makedirs(job_dir)

        self.props = ab_results(nxyz=self.nxyz,
                                charge=self.charge,
                                num_states=self.num_states,
                                surf=self.surfs[0],
                                job_dir=job_dir,
                                grad_config=self.grad_config,
                                nacv_config=self.nacv_config,
                                grad_details=self.grad_details,
                                nacv_details=self.nacv_details)
        self.correct_phase(old_force_nacv=old_force_nacv)
        self.step_num += 1

    def get_vel(self):
        """
        Velocities are in a.u. here, not ASE units
        """
        vel = np.stack([atoms.get_velocities()
                        for atoms in self.atoms_list])

        return vel

    def restart(self):
        super().restart()
        self.step_num = int(self.t / self.dt) + 2

    def new_force_calc(self):
        """
        Extra force calc on new state after hop
        """

        surf = self.surfs[0]
        needs_calc = np.bitwise_not(
            np.isfinite(
                self.forces[0, surf]
            )
        ).any()

        if not needs_calc:
            return

        new_job_dir = os.path.join(os.getcwd(),
                                   f"{self.step_num - 1}_extra")

        if os.path.isdir(new_job_dir):
            shutil.rmtree(new_job_dir)
        else:
            os.makedirs(new_job_dir)

        props = ab_results(nxyz=self.nxyz,
                           charge=self.charge,
                           num_states=self.num_states,
                           surf=surf,
                           job_dir=new_job_dir,
                           grad_config=self.grad_config,
                           nacv_config=self.nacv_config,
                           grad_details=self.grad_details,
                           nacv_details=self.nacv_details,
                           calc_nacv=False)

        key = f'energy_{surf}_grad'
        self.props[key] = props[key]

    def run(self):
        steps = math.ceil((self.max_time - self.t) / self.dt)
        if self.step_num == 0:
            self.update_props()

        for _ in range(steps):
            # if just hopped to new state, then we need to do a force
            # calculation on the new state too
            self.new_force_calc()
            self.save()
            self.step(needs_nbrs=False)

        with open(self.log_file, 'a') as f:
            f.write('\nTully surface hopping terminated normally.')

    @classmethod
    def from_file(cls,
                  file):

        all_params = load_params(file)
        atoms_list = make_atoms(all_params)

        instance = cls(atoms_list=atoms_list,
                       **all_params)

        return instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file',
                        type=str,
                        help='Info file with parameters',
                        default='job_info.json')
    args = parser.parse_args()

    path = args.params_file
    ab_tully = AbTully.from_file(path)

    try:
        ab_tully.run()
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()


if __name__ == '__main__':
    main()
