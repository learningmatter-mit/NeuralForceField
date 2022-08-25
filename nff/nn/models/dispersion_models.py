"""
Models with added empirical dispersion on top
"""

import torch
import numpy as np
from torch import nn

from nff.nn.models.painn import add_stress
from nff.utils.scatter import compute_grad
from nff.utils import constants as const
from nff.utils.dispersion import get_dispersion as base_dispersion
from nff.nn.models.painn import Painn

class PainnDispersion(nn.Module):

    def __init__(self,
                 modelparams,
                 painn_model=None):
        """
        `modelparams` has the same keys as in a regular PaiNN model, plus
        the required keys "functional" and "disp_type" for the added dispersion.

        You can also supply an existing PaiNN model instead of instantiating it from
        `modelparams`.
        """

        super().__init__()

        self.functional = modelparams["functional"]
        self.disp_type = modelparams["disp_type"]

        if painn_model is not None:
            self.painn_model = painn_model
        else:
            self.painn_model = Painn(modelparams=modelparams)

    def get_dispersion(self,
                       batch,
                       xyz):
        
        e_disp, r_ij_T, nbrs_T = base_dispersion(batch=batch,
                                                 xyz=xyz,
                                                 disp_type=self.disp_type,
                                                 functional=self.functional,
                                                 nbrs=batch.get('mol_nbrs'),
                                                 mol_idx=batch.get('mol_idx'))
        
        # convert to kcal / mol
        e_disp = e_disp * const.HARTREE_TO_KCAL_MOL

        return e_disp, r_ij_T, nbrs_T

    def run(self,
            batch,
            xyz=None,
            requires_stress=False,
            inference=False):

        # Normal painn stuff, part 1

        atomwise_out, xyz, r_ij, nbrs = self.painn_model.atomwise(batch=batch,
                                                                  xyz=xyz)

        if getattr(self.painn_model, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.painn_model.V_ex(r_ij, nbrs, xyz)
            atomwise_out['energy'] += r_ex

        all_results, xyz = self.painn_model.pool(batch=batch,
                                                 atomwise_out=atomwise_out,
                                                 xyz=xyz,
						                         r_ij=r_ij,
						                         nbrs=nbrs,
                                                 inference=inference)

        if requires_stress:
            all_results = add_stress(batch=batch,
                                     all_results=all_results,
                                     nbrs=nbrs,
                                     r_ij=r_ij)

        # add dispersion and gradients associated with it

        disp_grad = None
        e_disp, r_ij_T, nbrs_T = self.get_dispersion(batch=batch,
                                                     xyz=xyz)

        for key in self.painn_model.pool_dic.keys():
            # add dispersion energy
            if inference:
                add_e = e_disp.detach().cpu()
            else:
                add_e = e_disp

            all_results[key] = all_results[key] + add_e

            # add gradient for forces
            grad_key = "%s_grad" % key
            if grad_key in self.painn_model.grad_keys:
                if disp_grad is None:
                    disp_grad = compute_grad(inputs=xyz,
                                             output=e_disp)
                    if inference:
                        disp_grad = disp_grad.detach().cpu()

                all_results[grad_key] = all_results[grad_key] + disp_grad

        if requires_stress:
            # add gradient for stress
            disp_rij_grad = compute_grad(inputs=r_ij_T,
                                         output=e_disp)

            if batch['num_atoms'].shape[0] == 1:
                disp_stress_volume = torch.matmul(disp_rij_grad.t(), r_ij_T)
            else:
                allstress = []
                for j in range(batch['nxyz'].shape[0]):
                    allstress.append(
                        torch.matmul(
                            disp_rij_grad[torch.where(nbrs_T[:, 0] == j)].t(),
                            r_ij_T[torch.where(nbrs_T[:, 0] == j)]
                        )
                    )
                allstress = torch.stack(allstress)
                N = batch["num_atoms"].detach().cpu().tolist()
                split_val = torch.split(allstress, N)
                disp_stress_volume = torch.stack([i.sum(0)
                                                    for i in split_val])
            if inference:
                disp_stress_volume = disp_stress_volume.detach().cpu()
            all_results['stress_volume'] = all_results['stress_volume'] + \
                                                             disp_stress_volume

        # Normal painn stuff, part 2

        if getattr(self.painn_model, "compute_delta", False):
            all_results = self.painn_model.add_delta(all_results)

        return all_results, xyz

    def forward(self,
                batch,
                xyz=None,
                requires_stress=False,
                inference=False,
                **kwargs):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(batch=batch,
                              xyz=xyz,
                              requires_stress=requires_stress,
                              inference=inference)

        return results
