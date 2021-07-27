"""
Ab initio version of Tully's surface hopping
"""

import argparse
import os
import numpy as np

from nff.md.tully.dynamics import NeuralTully
from nff.md.tully.io import load_json, get_atoms

from nff.md.tully.ab_io import get_results as ab_results


"""
To-do:
- Sign tracking. Either force nacv delta/sigma or based on the
  signs of the orbital excitation amplitudes
- Putting htvs in the path so the parsers can be imported
"""


class AbTully(NeuralTully):
    def __init__(self,
                 charge,
                 grad_config,
                 nacv_config,
                 grad_details,
                 nacv_details,
                 **kwargs):

        NeuralTully.__init__(self,
                             **kwargs,
                             explicit_diabat_prop=False,
                             diabat_propagate=False,
                             simple_vel_scale=False,
                             max_gap_hop=float('inf'))

        self.charge = charge
        self.num_samples = 1

        self.grad_config = grad_config
        self.nacv_config = nacv_config
        self.grad_details = grad_details
        self.nacv_details = nacv_details

    def update_props(self,
                     step,
                     **kwargs):

        # how are we going to do the phase correction?

        job_dir = os.path.join(os.getcwd(), str(step))

        props = ab_results(nxyz=self.nxyz,
                           charge=self.charge,
                           old_U=self.U,
                           num_states=self.num_states,
                           surf=self.surfs[0],
                           job_dir=job_dir,
                           grad_config=self.grad_config,
                           nacv_config=self.nacv_config,
                           grad_details=self.grad_details,
                           nacv_details=self.nacv_details)

        self.props = props

    @classmethod
    def from_file(cls,
                  file):
        """
        - Need to read xyz and vel from json and convert to atoms
        objects (dumb but easiest)

        - Read charge from the json file 

        """

        all_params = load_json(file)

        vel = np.array(all_params['velocities'])
        atoms = get_atoms(all_params)

        atoms.set_velocities(vel)
        atoms_list = [atoms]

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
