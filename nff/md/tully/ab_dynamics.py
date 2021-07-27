"""
Ab initio version of Tully's surface hopping
"""

import argparse
import os
import math
import numpy as np
from ase import Atoms

from nff.md.tully.dynamics import (NeuralTully,
                                   TULLY_LOG_FILE,
                                   TULLY_SAVE_FILE)
from nff.md.tully.io import load_json, coords_to_xyz
from nff.md.tully.ab_io import get_results as ab_results
from nff.utils import constants as const


"""
To-do:

- Check the units on nacv and forces
- Sign tracking. Either force nacv delta/sigma or with a Q-Chem
  wavefunction overlap calculation (that would suck)

"""


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
        self.props = None
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

        self.log_template = self.setup_logging()
        self.p_hop = 0
        self.just_hopped = None
        self.explicit_diabat = False
        self.c = self.init_c()

        self.setup_save()
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

    @property
    def forces(self):
        inf = np.ones((self.num_atoms,
                       3)) * float('inf')
        _forces = np.stack([-self.props.get(f'energy_{i}_grad',
                                            inf).reshape(-1, 3)
                            for i in range(self.num_states)])
        _forces = _forces.reshape(1, *_forces.shape)

        return _forces

    def update_props(self,
                     *args,
                     **kwargs):

        job_dir = os.path.join(os.getcwd(), str(self.step_num))
        if not os.path.isdir(job_dir):
            os.makedirs(job_dir)

        props = ab_results(nxyz=self.nxyz,
                           charge=self.charge,
                           num_states=self.num_states,
                           surf=self.surfs[0],
                           job_dir=job_dir,
                           grad_config=self.grad_config,
                           nacv_config=self.nacv_config,
                           grad_details=self.grad_details,
                           nacv_details=self.nacv_details)

        self.props = props
        self.step_num += 1

    def get_vel(self):
        """
        Velocities are in a.u. here, not ASE units
        """
        vel = np.stack([atoms.get_velocities()
                        for atoms in self.atoms_list])

        return vel

    def run(self):
        steps = math.ceil(self.max_time / self.dt)
        self.update_props()

        for _ in range(steps):
            self.step(needs_nbrs=False)
            self.save()

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
