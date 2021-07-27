"""
Ab initio version of Tully's surface hopping
"""

import os

from nff.md.tully.dynamics import NeuralTully
# from nff.md.tully.io import load_json, get_atoms

from nff.md.tully.ab_io import get_results as ab_results


"""
To-do:
- Sign tracking
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

        pass

        # all_params = load_json(file)
        # atomsbatch = get_atoms(all_params)
        # ground_params = all_params['ground_params']

        # tully_params = all_params['tully_params']
        # model_path = os.path.join(all_params['weightpath'],
        #                           str(all_params["nnid"]))
        # tully_params.update({"model_path": model_path,
        #                      "device": all_params["device"],
        #                      "diabat_keys": all_params["diabat_keys"]})

        # instance = cls(atoms=atomsbatch,
        #                ground_params=ground_params,
        #                tully_params=tully_params)

        # return instance
