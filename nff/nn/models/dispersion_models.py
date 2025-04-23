"""
Models with added empirical dispersion on top
"""

import numpy as np
import torch
from torch import nn

from nff.nn.models.painn import Painn, PainnDiabat, add_stress
from nff.utils import constants as const
from nff.utils.dispersion import get_dispersion as base_dispersion
from nff.utils.dispersion import grimme_dispersion
from nff.utils.scatter import compute_grad


class PainnDispersion(nn.Module):
    def __init__(self, modelparams, painn_model=None):
        """
        `modelparams` has the same keys as in a regular PaiNN model, plus
        the required keys "functional" and "disp_type" for the added dispersion.

        You can also supply an existing PaiNN model instead of instantiating it from
        `modelparams`.
        """

        super().__init__()

        self.functional = modelparams["functional"]
        self.disp_type = modelparams["disp_type"]
        self.fallback_to_grimme = modelparams.get("fallback_to_grimme", True)

        if painn_model is not None:
            self.painn_model = painn_model
        else:
            self.painn_model = Painn(modelparams=modelparams)

    def get_dispersion(self, batch, xyz):
        e_disp, r_ij_T, nbrs_T = base_dispersion(
            batch=batch,
            xyz=xyz,
            disp_type=self.disp_type,
            functional=self.functional,
            nbrs=batch.get("mol_nbrs"),
            mol_idx=batch.get("mol_idx"),
        )
        # convert to kcal / mol
        e_disp = e_disp * const.HARTREE_TO_KCAL_MOL

        return e_disp, r_ij_T, nbrs_T

    def get_grimme_dispersion(self, batch, xyz):
        # all units are output in ASE units (eV and Angs)
        e_disp, stress_disp, forces_disp = grimme_dispersion(
            batch=batch, xyz=xyz, disp_type=self.disp_type, functional=self.functional
        )

        return e_disp, stress_disp, forces_disp

    def run(self, batch, xyz=None, requires_stress=False, grimme_disp=False, inference=False):
        # Normal painn stuff, part 1

        atomwise_out, xyz, r_ij, nbrs = self.painn_model.atomwise(batch=batch, xyz=xyz)

        if getattr(self.painn_model, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.painn_model.V_ex(r_ij, nbrs, xyz)
            for key in self.output_keys:
                atomwise_out[key] += r_ex

        all_results, xyz = self.painn_model.pool(
            batch=batch, atomwise_out=atomwise_out, xyz=xyz, r_ij=r_ij, nbrs=nbrs, inference=inference
        )

        if requires_stress:
            all_results = add_stress(batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij)

        # add dispersion and gradients associated with it

        disp_grad = None
        fallback_to_grimme = getattr(self, "fallback_to_grimme", True)

        if grimme_disp:
            e_disp, r_ij_T, nbrs_T = None
        else:
            e_disp, r_ij_T, nbrs_T = self.get_dispersion(batch=batch, xyz=xyz)

            for key in self.painn_model.pool_dic:
                # add dispersion energy
                add_e = e_disp.detach().cpu() if inference else e_disp

                # add gradient for forces
                grad_key = "%s_grad" % key
                if grad_key in self.painn_model.grad_keys:
                    if disp_grad is None:
                        disp_grad = compute_grad(inputs=xyz, output=e_disp)
                        if inference:
                            disp_grad = disp_grad.detach().cpu()

                    # check numerical stability of disp_grad pytorch calculation
                    if disp_grad.isnan().any() and fallback_to_grimme:
                        grimme_disp = True
                    else:
                        all_results[key] = all_results[key] + add_e
                        all_results[grad_key] = all_results[grad_key] + disp_grad

        if requires_stress and not grimme_disp:
            if e_disp is None or r_ij_T is None or nbrs_T is None:
                raise RuntimeError("Should not be reached, something went wrong")
            # add gradient for stress
            disp_rij_grad = compute_grad(inputs=r_ij_T, output=e_disp)

            if batch["num_atoms"].shape[0] == 1:
                disp_stress_volume = torch.matmul(disp_rij_grad.t(), r_ij_T)
            else:
                allstress = torch.stack([
                    torch.matmul(
                        disp_rij_grad[torch.where(nbrs_T[:, 0] == j)].t(), r_ij_T[torch.where(nbrs_T[:, 0] == j)]
                    )
                    for j in range(batch["nxyz"].shape[0])
                ])
                N = batch["num_atoms"].detach().cpu().tolist()
                split_val = torch.split(allstress, N)
                disp_stress_volume = torch.stack([i.sum(0) for i in split_val])
            if inference:
                disp_stress_volume = disp_stress_volume.detach().cpu()

            # check numerical stability of disp_grad pytorch calculation
            if disp_stress_volume.isnan().any() and fallback_to_grimme:
                grimme_disp = True
            else:
                all_results["stress_volume"] = all_results["stress_volume"] + disp_stress_volume

        # if there was numerical instability with disp_grad pytorch
        # re-calculate everything with Grimme dispersion instead
        # requires dftd3 executable
        if grimme_disp:
            e_disp, stress_disp, forces_disp = self.get_grimme_dispersion(batch=batch, xyz=xyz)
            all_results["e_disp"] = e_disp
            all_results["stress_disp"] = stress_disp
            all_results["forces_disp"] = forces_disp

        # Normal painn stuff, part 2

        if getattr(self.painn_model, "compute_delta", False):
            all_results = self.painn_model.add_delta(all_results)

        return all_results, xyz

    def forward(self, batch, xyz=None, requires_stress=False, grimme_disp=False, inference=False, **kwargs):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(
            batch=batch, xyz=xyz, requires_stress=requires_stress, grimme_disp=grimme_disp, inference=inference
        )

        return results


class PainnDiabatDispersion(PainnDiabat):
    def __init__(self, modelparams):
        super().__init__(modelparams=modelparams)
        self.functional = modelparams["functional"]
        self.disp_type = modelparams["disp_type"]

    def forward(
        self,
        batch,
        xyz=None,
        add_nacv=True,
        add_grad=True,
        add_gap=True,
        add_u=False,
        inference=False,
        do_nan=True,
        en_keys_for_grad=None,
    ):
        # get diabatic results
        results = super().forward(
            batch=batch,
            xyz=xyz,
            add_nacv=add_nacv,
            add_grad=add_grad,
            add_gap=add_gap,
            add_u=add_u,
            inference=inference,
            do_nan=do_nan,
            en_keys_for_grad=en_keys_for_grad,
        )
        xyz = results["xyz"]

        # get dispersion energy (I couldn't figure out how to sub-class
        # PainnDiabatDispersion with PainnDispersion without getting errors,
        # unless I put it before PainnDiabat, which isn't what I want. So
        # instead I just copied the logic for getting the disperson energy)
        e_disp, _, _ = base_dispersion(
            batch=batch,
            xyz=xyz,
            disp_type=self.disp_type,
            functional=self.functional,
            nbrs=batch.get("mol_nbrs"),
            mol_idx=batch.get("mol_idx"),
        )
        # convert to kcal / mol
        e_disp = e_disp * const.HARTREE_TO_KCAL_MOL

        # add dispersion energies to diabatic diagonals and adiabatic energies

        diabat_keys = self.diabatic_readout.diabat_keys
        diagonal_diabat_keys = np.diag(np.array(diabat_keys)).tolist()
        # don't just do self.diabatic_readout.energy_keys, because if we
        # have three diabatic states but only specify energy_keys =["energy_0",
        # "energy_1"], we won't have updated "energy_2" properly
        energy_keys = ["energy_%d" % i for i in range(len(diabat_keys))]

        for key in diagonal_diabat_keys + energy_keys:
            results[key] = results[key] + e_disp.reshape(results[key].shape)

        # add dispersion grads to diabatic diagonal gradients and
        # adiabatic gradients

        disp_grad = compute_grad(inputs=xyz, output=e_disp)

        grad_keys = [key + "_grad" for key in (diagonal_diabat_keys + energy_keys)]
        for key in grad_keys:
            if key in results:
                results[key] = results[key] + disp_grad.reshape(results[key].shape)

        return results
