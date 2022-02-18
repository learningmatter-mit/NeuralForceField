import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential
import copy

from nff.nn.layers import StochasticIncrease
from nff.utils.scatter import compute_grad
from nff.nn.layers import (Diagonalize, PainnRadialBasis, Dense,
                           Gaussian)
from nff.nn.tensorgrad import general_batched_hessian
from nff.utils.tools import layer_types
from nff.utils import constants as const


class DiabaticReadout(nn.Module):
    def __init__(self,
                 diabat_keys,
                 grad_keys,
                 energy_keys,
                 others_to_eig=None,
                 delta=False,
                 stochastic_dic=None,
                 cross_talk_dic=None,
                 hellmann_feynman=True):

        nn.Module.__init__(self)

        self.diag = Diagonalize()
        self.diabat_keys = diabat_keys
        self.grad_keys = grad_keys
        self.energy_keys = energy_keys
        self.delta = delta
        self.stochastic_modules = self.make_stochastic(stochastic_dic)
        self.cross_talk = self.make_cross_talk(cross_talk_dic)
        self.hf = hellmann_feynman
        self.others_to_eig = others_to_eig

    def make_cross_talk(self, cross_talk_dic):
        if cross_talk_dic is None:
            return

        cross_talk = CrossTalk(diabat_keys=self.diabat_keys,
                               energy_keys=self.energy_keys,
                               modes=cross_talk_dic["modes"],
                               pool_method=cross_talk_dic["pool_method"])
        return cross_talk

    def make_stochastic(self, stochastic_dic):
        """
        E.g. stochastic_layers = {"energy_1": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.5},
                                "lam": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.25}
                                    },
                                 "d1": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.5}}

        For energy_1 it's understood that the adiabatic gap between state 1
        and state 0 will be increased. Similarly for d1 is's understood that
        the diabatic gap between state 1 and state 0 will be increased. If we
        had also specified, for example, energy_2 and d2, then those gaps
        will be increased relative to the new values of energy_1 and d_1.

        For "lam", an off-diagonal diabatic element, it's understood only that
        its magnitude will decrease.
        """

        stochastic_modules = nn.ModuleDict({})
        if stochastic_dic is None:
            return stochastic_modules

        for key, layer_dic in stochastic_dic.items():
            if layer_dic["name"].lower() == "stochasticincrease":
                params = layer_dic["param"]
                layer = StochasticIncrease(**params)
                stochastic_modules[key] = layer
            else:
                raise NotImplementedError

        return stochastic_modules

    def quant_to_mat(self,
                     num_atoms,
                     results,
                     diabat_keys):
        num_states = len(diabat_keys)

        test_result = results[diabat_keys[0][0]]
        shape = test_result.shape
        device = test_result.device

        mat = torch.zeros(*shape,
                          num_states,
                          num_states).to(device)

        for i in range(num_states):
            for j in range(num_states):
                diabat_key = diabat_keys[i][j]
                mat[..., i, j] = results[diabat_key]

        return mat

    def results_to_dmat(self, results, num_atoms):

        d_mat = self.quant_to_mat(num_atoms=num_atoms,
                                  results=results,
                                  diabat_keys=self.diabat_keys)

        return d_mat

    def compute_eig(self,
                    d_mat,
                    train):

        dim = d_mat.shape[-1]
        # do analytically if possible to avoid sign ambiguity
        # in the eigenvectors, which leads to worse training
        # results for the nacv
        if dim == 2:
            ad_energies, u = self.diag(d_mat)
        # otherwise do numerically
        else:
            # catch any nans - if you leave them before
            # calculating the eigenvectors then they will
            # raise an error
            nan_idx = torch.bitwise_not(torch.isfinite(d_mat)).any(-1).any(-1)

            if not train:
                d_mat[nan_idx, :, :] = 0

            # diagonalize
            ad_energies, u = torch.symeig(d_mat, True)

        return ad_energies, u, nan_idx

    def add_diag(self,
                 results,
                 num_atoms,
                 train):

        d_mat = self.results_to_dmat(results, num_atoms)
        # ad_energies, u = torch.symeig(d_mat, True)
        ad_energies, u, nan_idx = self.compute_eig(d_mat,
                                                   train=train)
        # results.update({key: ad_energies[:, i].reshape(-1, 1)
        #                 for i, key in enumerate(self.energy_keys)})

        results.update({f'energy_{i}': ad_energies[:, i].reshape(-1, 1)
                        for i in range(ad_energies.shape[1])})

        return results, u, nan_idx

    def get_diabat_grads(self,
                         results,
                         xyz,
                         num_atoms,
                         inference):

        num_states = len(self.diabat_keys)
        total_atoms = sum(num_atoms)
        diabat_grads = torch.zeros(
            num_states,
            num_states,
            total_atoms,
            3
        ).to(xyz.device)

        for i in range(num_states):
            for j in range(num_states):
                diabat_key = self.diabat_keys[i][j]
                grad_key = diabat_key + "_grad"
                if grad_key in results:
                    grad = results[grad_key]
                else:
                    grad = compute_grad(inputs=xyz,
                                        output=results[diabat_key],
                                        allow_unused=True)

                if inference:
                    grad = grad.detach()
                results[grad_key] = grad
                diabat_grads[i, j, :, :] = grad
        return results, diabat_grads

    def add_all_grads(self,
                      xyz,
                      results,
                      num_atoms,
                      u,
                      inference):

        results, diabat_grads = self.get_diabat_grads(results=results,
                                                      xyz=xyz,
                                                      num_atoms=num_atoms,
                                                      inference=inference)
        split_grads = torch.split(diabat_grads,
                                  num_atoms, dim=2)

        add_keys = []

        for k, this_grad in enumerate(split_grads):
            this_u = u[k]
            ad_grad = torch.einsum('ki, klnm, lj -> ijnm',
                                   this_u, this_grad, this_u)

            num_states = ad_grad.shape[0]
            for i in range(num_states):
                for j in range(num_states):
                    key = (f"energy_{i}_grad" if (i == j)
                           else f"force_nacv_{i}{j}")
                    if key not in results:
                        results[key] = []
                        add_keys.append(key)
                    results[key].append(ad_grad[i, j])

                    if i != j:
                        if not all([f"energy_{i}" in results,
                                    f"energy_{j}" in results]):
                            continue
                        gap = results[f"energy_{j}"] - results[f"energy_{i}"]
                        nacv = ad_grad[i, j] / gap[k]
                        nacv_key = f"nacv_{i}{j}"
                        if nacv_key not in results:
                            results[nacv_key] = []
                            add_keys.append(nacv_key)
                        results[nacv_key].append(nacv)

        for key in add_keys:
            results[key] = torch.cat(results[key])

        return results

    def quants_to_eig(self,
                      num_atoms,
                      results,
                      u):
        """
        Convert other quantities (e.g. diabatic dipole
        moments) into the adiabatic basis.

        Args:
            others_to_eig (list[list]): list of the form
                [
                    [[quantA_00, quantA_01], [quantA_01, quantA_11]],
                    [[quantB_00, quantB_01], [quantB_01, quantB_11]]
                ]
        """

        for other_keys in self.others_to_eig:

            # mat has shape num_mols x num_states x num_states for a scalar,
            # and num_mols x 3 x num_states x num_states for vector

            mat = self.quant_to_mat(num_atoms=num_atoms,
                                    results=results,
                                    diabat_keys=other_keys)
            # u has shape num_mols x num_states x num_states

            to_eig = torch.einsum('nki, n...kl, nlj -> n...ij',
                                  u, mat, u)

            base_key = other_keys[0][0].split("_")[0]
            num_states = len(other_keys)

            for i in range(num_states):
                for j in range(num_states):
                    if j < i:
                        continue
                    if i == j:
                        key = f"{base_key}_{i}"
                    else:
                        key = f"trans_{base_key}_{i}{j}"
                    results[key] = to_eig[..., i, j]

            return results

    def add_adiabat_grads(self,
                          xyz,
                          results,
                          inference,
                          en_keys_for_grad):

        if en_keys_for_grad is None:
            en_keys_for_grad = copy.deepcopy(self.energy_keys)

        for key in en_keys_for_grad:
            val = results[key]
            grad = compute_grad(inputs=xyz,
                                output=val,
                                allow_unused=True)
            if inference:
                grad = grad.detach()
            results[key + "_grad"] = grad

        return results

    def add_gap(self,
                results,
                add_grad):

        # diabatic gap

        bottom_key = self.diabat_keys[0][0]
        top_key = self.diabat_keys[1][1]
        gap = results[top_key] - results[bottom_key]
        results.update({"abs_diabat_gap": abs(gap)})

        # adiabatic gap

        num_states = len(self.energy_keys)
        for i in range(num_states):
            for j in range(num_states):
                if j <= i:
                    continue

                upper_key = self.energy_keys[j]
                lower_key = self.energy_keys[i]
                gap = results[upper_key] - results[lower_key]
                results.update({f"{upper_key}_{lower_key}_delta": gap})

                if add_grad:
                    upper_grad_key = upper_key + "_grad"
                    lower_grad_key = lower_key + "_grad"

                    grad_keys = [upper_grad_key, lower_grad_key]
                    if not all([i in results for i in grad_keys]):
                        continue

                    gap_grad = results[upper_grad_key] - \
                        results[lower_grad_key]
                    results.update({f"{upper_key}_{lower_key}_delta_grad":
                                    gap_grad})

        return results

    def add_stochastic(self, results):

        # any deltas that you want to decrease, whether adiabatic
        # or diabatic

        module_keys = self.stochastic_modules.keys()
        diabat_keys = np.array(self.diabat_keys)
        diag_diabat_keys = np.array(self.diabat_keys).diagonal()
        num_states = diabat_keys.shape[0]

        diag_adiabat_incr = [i for i in module_keys if i
                             not in diabat_keys.reshape(-1)]
        odiag_diabat_incr = []
        diag_diabat_incr = []

        for i in range(num_states):
            for j in range(num_states):
                key = diabat_keys[i, j]
                if key not in module_keys:
                    continue
                if i == j:
                    diag_diabat_incr.append(key)
                else:
                    odiag_diabat_incr.append(key)

        odiag_diabat_incr = list(set(odiag_diabat_incr))
        # all diagonals, both adiabatic and diabatic
        all_diag_incr = list(set(diag_adiabat_incr + diag_diabat_incr))

        # directly scale off-diagonal diabatic elements
        for key in odiag_diabat_incr:
            output = results[key]
            results[key] = self.stochastic_modules[key](output)

        # sequentially increase gap between adiabatic and diagonal
        # diabatic elements

        for diag_keys in [diag_diabat_keys, self.energy_keys]:
            for i, key in enumerate(diag_keys):
                # start with first excited state, meaning you ignore
                # the lowest key
                if i == 0:
                    continue
                # don't do anything if we didn't ask it to be scaled
                if key not in all_diag_incr:
                    continue

                lower_key = diag_diabat_keys[i - 1]
                delta = results[key] - results[lower_key]

                # stochastically increase the difference between the
                # two states
                change = -delta + self.stochastic_modules[key](delta)
                results[key] = results[key] + change

        return results

    def idx_to_grad_idx(self,
                        num_atoms,
                        nan_idx):

        if isinstance(num_atoms, torch.Tensor):
            atom_tens = num_atoms
        else:
            atom_tens = torch.LongTensor(num_atoms)

        end_idx = torch.cumsum(atom_tens, dim=0)
        start_idx = torch.cat([torch.tensor([0]).to(end_idx.device),
                               end_idx[:-1]])
        start_end = torch.stack([start_idx, end_idx], dim=-1)
        nan_start_end = start_end[nan_idx]

        return nan_start_end

    def get_all_keys(self):

        num_states = len(self.diabat_keys)
        unique_diabats = list(
            set(
                np.array(self.diabat_keys)
                .reshape(-1).tolist()
            )
        )
        adiabat_keys = [f'energy_{i}' for i in range(num_states)]

        en_keys = unique_diabats + adiabat_keys
        grad_keys = [i + "_grad" for i in en_keys]
        grad_keys += [f"force_nacv_{i}{j}" for i in range(num_states)
                      for j in range(num_states)]
        grad_keys += [f"nacv_{i}{j}" for i in range(num_states)
                      for j in range(num_states)]

        return en_keys, grad_keys

    def add_nans(self,
                 batch,
                 results,
                 nan_idx,
                 train):

        en_keys, grad_keys = self.get_all_keys()
        nan_grad_idx = self.idx_to_grad_idx(num_atoms=batch['num_atoms'],
                                            nan_idx=nan_idx)

        for key in [*en_keys, 'U']:
            if key not in results:
                continue

            if train:
                continue

            results[key][nan_idx] *= float('nan')

        for key in grad_keys:
            if key not in results:
                continue
            for idx in nan_grad_idx:
                if train:
                    continue
                results[key][idx[0]: idx[1]] = (results[key][idx[0]: idx[1]]
                                                * float('nan'))

    def forward(self,
                batch,
                xyz,
                results,
                add_nacv=False,
                add_grad=True,
                add_gap=True,
                add_u=False,
                inference=False,
                do_nan=True,
                en_keys_for_grad=None):

        if not hasattr(self, "delta"):
            self.delta = False

        if not hasattr(self, "stochastic_modules"):
            self.stochastic_modules = nn.ModuleDict({})

        if not hasattr(self, "cross_talk"):
            self.cross_talk = None

        num_atoms = batch["num_atoms"].detach().cpu().tolist()

        # preprocessing applied to diabats before computing
        # adiabats

        if self.delta:
            diag_diabat = np.array(self.diabat_keys).diagonal()
            for key in diag_diabat[1:]:
                results[key] += results[diag_diabat[0]]

        if self.cross_talk:
            results = self.cross_talk(results)

        # calculation of adiabats and their gradients

        results, u, nan_idx = self.add_diag(results=results,
                                            num_atoms=num_atoms,
                                            train=(not inference))

        if add_u:
            results["U"] = u

        if add_grad and add_nacv:
            results = self.add_all_grads(xyz=xyz,
                                         results=results,
                                         num_atoms=num_atoms,
                                         u=u,
                                         inference=inference)
        elif add_grad:
            results = self.add_adiabat_grads(xyz=xyz,
                                             results=results,
                                             inference=inference,
                                             en_keys_for_grad=en_keys_for_grad)

        # add back any nan's that were originally set to zeros
        # to avoid an error in diagonalization

        if do_nan:
            self.add_nans(batch=batch,
                          results=results,
                          nan_idx=nan_idx,
                          train=(not inference))

        if getattr(self, "others_to_eig", None):
            results = self.quants_to_eig(num_atoms=num_atoms,
                                         results=results,
                                         u=u)

        if add_gap:
            results = self.add_gap(results, add_grad=add_grad)

        if self.training:
            results = self.add_stochastic(results)

        return results


class GapCouplingProduct(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 coupling_key,
                 diag_keys,
                 **kwargs):
        """
        Args:
            n_rbf (int): number of basis functions for gap
                embedding
            cutoff (float): max gap, in eV. Anything higher
                gives a multiplier of 1.
        """
        super().__init__()

        basis = PainnRadialBasis(n_rbf=n_rbf,
                                 cutoff=cutoff,
                                 learnable_k=False)

        activation = Gaussian(mean=0,
                              sigma=0.1,
                              learnable_mean=False,
                              learnable_sigma=False,
                              normalize=False)

        dense = Dense(in_features=n_rbf,
                      out_features=1,
                      bias=False,
                      activation=activation)

        self.gap_network = Sequential(basis, dense)
        self.coupling_key = coupling_key
        self.diag_keys = diag_keys

    def forward(self, results):

        d1 = results[self.diag_keys[1]]
        d0 = results[self.diag_keys[0]]
        coupling = results[self.coupling_key]

        gap = (d1 - d0).abs()
        multiplier = self.gap_network(gap).reshape(-1)
        new_coupling = coupling * multiplier
        new_results = {self.coupling_key: new_coupling}

        return new_results


class ShiftedSigmoid(nn.Module):
    def __init__(self,
                 x_shift,
                 y_shift,
                 slope):

        super().__init__()

        self.y_shift = y_shift
        self.x_shift = x_shift
        self.slope = slope

    def forward(self, inp):
        norm_factor = 1 / (1 - self.y_shift)
        sig = 1 / (1 + torch.exp(-(inp - self.x_shift) * self.slope))
        out = norm_factor * (sig - self.y_shift)
        return out


class GapCouplingProductFixed(nn.Module):
    def __init__(self,
                 params,
                 coupling_key,
                 diag_keys,
                 **kwargs):
        """
        Example:
           params = {"name": "sigmoid",
                    "slope": 0.33,
                     "x_shift": 0.0,
                     "y_shift": 0.4}
        Note: everything in kcal
        """

        super().__init__()

        self.smoothing_func = self.get_smoothing(params)
        self.coupling_key = coupling_key
        self.diag_keys = diag_keys

    def get_smoothing(self, params):
        # don't let arbitrary activations be used
        # because we'll want to modify their parameters.
        # If we haven't implemented the modification
        # options then those modifications will be silently
        # ignored

        name = params["name"]

        if name.lower() == 'sigmoid':
            y_shift = params.get("y_shift", 0)
            x_shift = params.get("x_shift", 0)
            slope = params.get("slope", 1)

            smoothing_func = ShiftedSigmoid(x_shift=x_shift,
                                            y_shift=y_shift,
                                            slope=slope)
        else:
            raise NotImplementedError

        return smoothing_func

    def forward(self, results):

        d0 = results[self.diag_keys[0]]
        d1 = results[self.diag_keys[1]]
        coupling = results[self.coupling_key]

        gap = (d1 - d0).abs()
        scaling = self.smoothing_func(gap).reshape(-1)
        new_coupling = coupling * scaling

        results[self.coupling_key] = new_coupling

        return results


class GapCouplingConcat(nn.Module):
    def __init__(self,
                 num_hidden,
                 num_feat_layers,
                 activation,
                 feat_dim,
                 coupling_key,
                 diag_keys,
                 softplus=None,
                 beta=None,
                 **kwargs):

        super().__init__()

        feat_layers = []
        for layer in range(num_feat_layers):
            act_func = layer_types[activation]() if (
                layer != (num_feat_layers - 1)) else None
            in_features = 1 if (layer == 0) else num_hidden

            featurizer = Dense(in_features=in_features,
                               out_features=num_hidden,
                               bias=True,
                               activation=act_func)
            feat_layers.append(featurizer)

        self.featurizer = Sequential(*feat_layers)

        new_dim = (feat_dim + num_hidden)
        final_act = nn.Softplus(beta=beta) if softplus else None

        self.readout = Sequential(
            Dense(in_features=new_dim,
                  out_features=new_dim//2,
                  bias=True,
                  activation=layer_types[activation]()),

            # softplus to make the coupling always positive
            # and give the inputs lots of ways to
            # turn into 0

            Dense(in_features=new_dim//2,
                  out_features=1,
                  bias=True,
                  activation=final_act)
        )

        self.coupling_key = coupling_key
        self.diag_keys = diag_keys

    def forward(self, results):

        coupling = results[self.coupling_key]
        # this is a bit hacky and a waste of compute
        # but I'm not sure the best way to do it

        coupling.detach().cpu()
        results.pop(self.coupling_key)

        d1 = results[self.diag_keys[1]]
        d0 = results[self.diag_keys[0]]

        gap = (d1 - d0).abs().reshape(-1, 1)
        gap_feats = self.featurizer(gap)
        base_feats = results[f"{self.coupling_key}_features"]

        cat_feats = torch.cat([base_feats, gap_feats], dim=-1)
        coupling = self.readout(cat_feats).reshape(-1)
        results[self.coupling_key] = coupling

        return results


class GapCouplingProdConcat(nn.Module):
    def __init__(self,
                 num_hidden,
                 num_feat_layers,
                 activation,
                 feat_dim,
                 product_params,
                 coupling_key,
                 diag_keys,
                 **kwargs):
        super().__init__()

        self.concat_module = GapCouplingConcat(num_hidden=num_hidden,
                                               num_feat_layers=num_feat_layers,
                                               activation=activation,
                                               feat_dim=feat_dim,
                                               coupling_key=coupling_key,
                                               diag_keys=diag_keys)
        self.product_module = GapCouplingProductFixed(
            params=product_params,
            coupling_key=coupling_key,
            diag_keys=diag_keys)

    def forward(self, results):

        results = self.concat_module(results)
        results = self.product_module(results)

        return results


class CrossTalk(nn.Module):
    """
    Module for making diabatic matrix elements talk to
    each other
    """

    def __init__(self,
                 diabat_keys,
                 energy_keys,
                 modes,
                 pool_method):
        """
        Example:
            modes = {"gap_coupling_product": [
                            {
                                "params": {
                                            "n_rbf": 10,
                                            "cutoff": 11.53,
                                            "states": [0, 1]
                                          }
                            }
                        ],
                    "gap_coupling_concat": [
                            {
                                "params": {
                                            "num_hidden": 64,
                                            "num_feat_layers": 2,
                                            "activation": "swish",
                                            "feat_dim": 128,
                                            "states": [0, 1]
                                          }
                            }

                        ]

                    }
            diabat_keys = [["d0", "lam"], ["lam", "d1"]]
            energy_keys = ["energy_0", "energy_1"]
            pool_method = "mean"

        Note that everything is done in kcal/mol.

        """

        super().__init__()

        self.diabat_keys = np.array(diabat_keys)
        self.energy_keys = np.array(energy_keys)
        self.coupling_modules = self.make_modules(modes)
        self.pool_method = pool_method

    def make_modules(self, modes):
        modules = nn.ModuleList([])
        # cycle through different module types
        for name, dic_list in modes.items():
            # cycle through different implementations of
            # the module (e.g. gap coupling product between
            # different pairs of states)

            for dic in dic_list:
                params = dic["params"]
                states = np.array(params["states"])
                diag_keys = self.diabat_keys.diagonal()[states]
                coupling_key = self.diabat_keys[states[0], states[1]]

                if name.lower() == "gap_coupling_product":
                    module = GapCouplingProduct(coupling_key=coupling_key,
                                                diag_keys=diag_keys,
                                                **params)

                elif name.lower() == "gap_coupling_concat":

                    module = GapCouplingConcat(
                        coupling_key=coupling_key,
                        diag_keys=diag_keys,
                        **params
                    )
                elif name.lower() == "gap_coupling_prod_concat":

                    module = GapCouplingProdConcat(
                        coupling_key=coupling_key,
                        diag_keys=diag_keys,
                        **params)

                elif name.lower() == "gap_coupling_product_fixed":
                    module = GapCouplingProductFixed(params=params,
                                                     coupling_key=coupling_key,
                                                     diag_keys=diag_keys)

                else:
                    raise NotImplementedError

                modules.append(module)
        return modules

    def forward(self, results):
        combined_results = {}

        for module in self.coupling_modules:
            new_results = module(results)
            for key, new_val in new_results.items():
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].append(new_val)

        final_results = {}
        for key, lst in combined_results.items():
            stacked_val = torch.stack(lst)
            if self.pool_method == "mean":
                pool_val = stacked_val.mean(0)
            elif self.pool_method == "sum":
                pool_val = stacked_val.sum(0)
            else:
                raise NotImplementedError

            final_results[key] = pool_val

        for key, val in results.items():
            if key not in combined_results:
                final_results[key] = val

        return final_results


class AdiabaticReadout(nn.Module):
    def __init__(self,
                 output_keys,
                 grad_keys,
                 abs_name):

        super().__init__()

        self.abs_fn = self.get_abs(abs_name)
        self.output_keys = output_keys
        self.grad_keys = grad_keys

    def get_abs(self, abs_name):
        if abs_name == "abs":
            return abs
        elif abs_name is None:
            return lambda x: x
        elif abs_name in layer_types:
            return layer_types[abs_name]()
        else:
            raise NotImplementedError

    def forward(self,
                results,
                xyz):

        ordered_keys = sorted(self.output_keys, key=lambda x:
                              int(x.split("_")[-1]))

        for i, key in enumerate(ordered_keys):
            if i == 0:
                continue
            lower_key = ordered_keys[i - 1]
            results[key] = results[lower_key] + self.abs_fn(results[key])

        for key in self.grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz,
                                allow_unused=True)
            results[key] = grad

        return results


class AdiabaticNacv(nn.Module):
    def __init__(self,
                 model,
                 coupled_states=None,
                 gap_threshold=0.5):

        super().__init__()
        self.model = model
        self.output_keys = self.model.output_keys
        self.grad_keys = self.model.grad_keys

        self.coupled_states = self.get_coupled_states(coupled_states)
        self.delta_keys = self.get_delta_keys()
        self.gap_threshold = gap_threshold * const.EV_TO_KCAL_MOL

    def get_coupled_states(self,
                           coupled_states):
        if coupled_states is not None:
            return coupled_states

        coupled_states = []
        for i, e_i in enumerate(self.output_keys):
            for j, e_j in enumerate(self.output_keys):
                if j >= i:
                    continue
                needs_grad = [key + "_grad" in self.grad_keys
                              for key in [e_i, e_j]]

                if all(needs_grad):
                    coupled_states.append([i, j])
        return coupled_states

    def get_delta_keys(self):
        keys = []
        for i, e_i in enumerate(self.model.output_keys):
            for j, e_j in enumerate(self.model.output_keys):
                if j >= i:
                    continue
                if [i, j] not in self.coupled_states:
                    continue
                key = f"{e_i}_{e_j}_delta"
                keys.append(key)
        return keys

    def process_hess(self,
                     batch,
                     results,
                     states):

        e_i = self.output_keys[min(states)]
        e_j = self.output_keys[max(states)]

        key = f"{e_j}_{e_i}_delta"
        hess_key = key + "_hess"

        gap = results[key]
        gap_hess = results[hess_key]

        # need to do a singular value decomposition to get rid
        # of the zero eigenvalues too

        # Also need to just predict 0 if the gap is too high

        return gap, gap_hess

    def nacv_from_hess(self,
                       batch,
                       results,
                       states):

        gaps, gap_hess = self.process_hess(batch=batch,
                                           results=results,
                                           states=states)
        nacvs = []

        for gap, hess in zip(gaps, gap_hess):

            # Return 0 if the gap is > threshold
            # This is very inefficient right now because the
            # Hessian is computed even if it doesn't get used

            dim = int((hess.reshape(-1).shape[0]) ** 0.5)

            if gap > self.gap_threshold:
                nacv = (torch.zeros(dim // 3, 3)
                        .to(gap.device))
                nacvs.append(nacv)
                continue

            # if you simplify eqs. 5 and 6 in the SchNarc
            # paper you get this

            nacv_kron = (1 / 4 * gap * hess).reshape(dim, dim)

            eigs, vecs = torch.symeig(nacv_kron, True)
            argmax = eigs.argmax()
            lam = eigs[argmax]
            v = vecs[:, argmax]
            nacv = v * (lam ** 0.5)

            nacvs.append(nacv.reshape(-1, 3))

        nacvs = torch.cat(nacvs)

        return nacvs

    def nacv_add(self,
                 batch,
                 results):

        for states in self.coupled_states:
            nacv = self.nacv_from_hess(batch=batch,
                                       results=results,
                                       states=states)
            i = states[0]
            j = states[1]
            key = f"force_nacv_{i}{j}"

            results[key] = nacv

        return results

    def fix_pad(self,
                batch,
                results):
        num_atoms = batch['num_atoms']
        if not isinstance(num_atoms, list):
            num_atoms = num_atoms.tolist()

        max_atoms = max(num_atoms)
        pad_num_atoms = [max_atoms] * len(num_atoms)

        for key in self.grad_keys:
            real = []
            split = list(torch.split(results[key],
                                     pad_num_atoms))
            for val, num in zip(split, num_atoms):
                real.append(val[:num])
            real = torch.cat(real)

            results[key] = real

        return results

    def forward(self,
                batch,
                report_hess=False,
                **kwargs):

        device = batch['nxyz'].device
        results = general_batched_hessian(batch=batch,
                                          keys=self.delta_keys,
                                          device=device,
                                          model=self.model,
                                          forward=None,
                                          **kwargs)
        results = self.nacv_add(batch=batch,
                                results=results)
        results = self.fix_pad(batch=batch,
                               results=results)

        if not report_hess:
            for key in list(results.keys()):
                if 'hess' in key:
                    results.pop(key)

        return results
