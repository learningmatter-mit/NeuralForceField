import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential

from nff.nn.layers import StochasticIncrease
from nff.utils.scatter import compute_grad
from nff.nn.layers import (Diagonalize, PainnRadialBasis, Dense,
                           Gaussian)
from nff.utils.tools import layer_types


class DiabaticReadout(nn.Module):
    def __init__(self,
                 diabat_keys,
                 grad_keys,
                 energy_keys,
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

    def get_hf_nacv(self,
                    U,
                    xyz,
                    N,
                    results):

        num_states = U.shape[2]
        split_xyz = torch.split(xyz, N)

        h_grads = [torch.zeros(num_states, num_states,
                               this_xyz.shape[0], 3).to(xyz.device)
                   for this_xyz in split_xyz]

        for i in range(num_states):
            for j in range(num_states):
                h_ij = results[self.diabat_keys[i][j]]
                this_grad = compute_grad(inputs=xyz,
                                         output=h_ij)

                grad_split = torch.split(this_grad, N)
                for k, grad in enumerate(grad_split):
                    h_grads[k][i, j, :, :] = grad

        nacvs = []
        force_nacvs = []

        for k, h_grad in enumerate(h_grads):
            this_u = U[k]
            this_u_dag = this_u.transpose(0, 1)
            force_nacv = torch.einsum('ik, klnm, lj -> ijnm',
                                      this_u_dag, h_grad, this_u)

            gaps = torch.zeros(num_states, num_states).to(force_nacv.device)
            for i in range(num_states):
                for j in range(num_states):
                    en_i = results[f'energy_{i}'][k]
                    en_j = results[f'energy_{j}'][k]
                    gaps[i, j] = en_j - en_i

            gaps = gaps.reshape(num_states, num_states, 1, 1)
            nacv = force_nacv / gaps

            nacvs.append(nacv)
            force_nacvs.append(force_nacv)

        nacv_cat = torch.cat(nacvs, axis=-2)
        force_nacv_cat = torch.cat(force_nacvs, axis=-2)

        return nacv_cat, force_nacv_cat

    def get_non_hf_nacv(self,
                        U,
                        xyz,
                        N,
                        results):

        num_states = U.shape[2]
        split_xyz = torch.split(xyz, N)

        u_grads = [torch.zeros(num_states, num_states,
                               this_xyz.shape[0], 3).to(xyz.device)
                   for this_xyz in split_xyz]

        for i in range(num_states):
            for j in range(num_states):
                this_grad = compute_grad(inputs=xyz,
                                         output=U[:, i, j])

                grad_split = torch.split(this_grad, N)
                for k, grad in enumerate(grad_split):
                    u_grads[k][i, j, :, :] = grad

        # m, l, and s are state indices that get summed out
        # i and j are state indices that don't get summed out
        # a = N_at is the number of atoms
        # t = 3 is the number of directions for each atom

        nacvs = []
        force_nacvs = []

        for k, u_grad in enumerate(u_grads):
            this_u = U[k]
            nacv = torch.einsum('ki, kjnm -> ijnm',
                                this_u, u_grad)

            gaps = torch.zeros(num_states, num_states).to(nacv.device)
            for i in range(num_states):
                for j in range(num_states):
                    en_i = results[f'energy_{i}'][k]
                    en_j = results[f'energy_{j}'][k]
                    gaps[i, j] = en_j - en_i

            gaps = gaps.reshape(num_states, num_states, 1, 1)
            force_nacv = nacv * gaps

            nacvs.append(nacv)
            force_nacvs.append(force_nacv)

        nacv_cat = torch.cat(nacvs, axis=-2)
        force_nacv_cat = torch.cat(force_nacvs, axis=-2)

        return nacv_cat, force_nacv_cat

    def get_nacv(self,
                 U,
                 xyz,
                 N,
                 results):
        """
        hf (bool): whether to use Hellman-Feynman
        """

        if not hasattr(self, "hf"):
            self.hf = True

        if self.hf:
            func = self.get_hf_nacv
        else:
            func = self.get_non_hf_nacv
        nacv_cat, force_nacv_cat = func(U=U,
                                        xyz=xyz,
                                        N=N,
                                        results=results)

        return nacv_cat, force_nacv_cat

    def add_nacv(self, results, u, xyz, N):

        nacv, force_nacv = self.get_nacv(U=u,
                                         xyz=xyz,
                                         N=N,
                                         results=results)
        num_states = nacv.shape[0]
        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    continue
                this_nacv = nacv[i, j, :, :]
                this_force_nacv = force_nacv[i, j, :, :]

                results[f"nacv_{i}{j}"] = this_nacv
                results[f"force_nacv_{i}{j}"] = this_force_nacv

        return results

    def add_diag(self,
                 results,
                 N,
                 xyz,
                 add_nacv):

        diabat_keys = np.array(self.diabat_keys)
        dim = diabat_keys.shape[0]
        num_geoms = len(N)
        diabat_ham = (torch.zeros(num_geoms, dim, dim)
                      .to(xyz.device))
        for i in range(dim):
            for j in range(dim):
                key = diabat_keys[i, j]
                diabat_ham[:, i, j] = results[key]

        ad_energies, u = self.diag(diabat_ham)

        results.update({key: ad_energies[:, i].reshape(-1, 1)
                        for i, key in enumerate(self.energy_keys)})
        if add_nacv:
            results = self.add_nacv(results=results,
                                    u=u,
                                    xyz=xyz,
                                    N=N)

        return results

    def choose_grad_route(self, extra_grads):
        """
        If gradients of certain diabatic states are asked for, then
        decide the most efficient way to calculate both those and
        the adiabatic gradients.
        """

        # unique diabatic quantities
        unique_diabats = list(set(np.array(self.diabat_keys)
                                  .reshape(-1).tolist()))

        # extra quantities whose gradients were requested
        extra_quants = [i.replace("_grad", "") for i in
                        np.array(extra_grads).reshape(-1).tolist()]

        # diabatic keys for which gradients were requested
        diabats_need_grad = list(set([i for i in extra_quants
                                      if i in unique_diabats]))

        # adiabatic energies for which gradients were requested
        energies_need_grad = [i.replace("_grad", "") for i in self.grad_keys
                              if i.replace("_grad", "") in self.energy_keys]

        # number of diabatic gradients needed to make sure we get all the
        # adiabatic gradients right

        num_diabat_to_en = len(unique_diabats)

        # number of gradients needed if we compute all gradients separately

        num_separate = len(diabats_need_grad) + len(energies_need_grad)

        # choose the route that takes fewer calculations

        if num_diabat_to_en < num_separate:
            route = "diabat_to_en"
        else:
            route = "separate"

        return route

    def compute_diabat_grad(self,
                            results,
                            xyz,
                            N):
        unique_diabats = list(set(np.array(self.diabat_keys)
                                  .reshape(-1).tolist()))

        grad_dic = {}
        for d_key in unique_diabats:
            grad = compute_grad(inputs=xyz, output=results[d_key])
            grad_dic[f"{d_key}_grad"] = grad

        # num_diabat x num_atoms x 3
        d_grads = torch.stack([grad_dic[d_key + "_grad"]
                               for d_key in unique_diabats])

        return grad_dic, d_grads, unique_diabats

    def compute_dE_dD(self,
                      results,
                      en_keys,
                      unique_diabats,
                      num_mols,
                      num_states):

        num_diabat = len(unique_diabats)
        device = results[en_keys[0]].device

        # num_mols x num_states x num_diabat
        dE_dD = torch.zeros(num_mols, num_states, num_diabat)

        for i, en_key in enumerate(en_keys):
            for j, d_key in enumerate(unique_diabats):
                grad = compute_grad(inputs=results[d_key],
                                    output=results[en_key])
                dE_dD[:, i, j] = grad

        return dE_dD

    def compute_all_grads(self,
                          results,
                          xyz,
                          N):
        """
        Compute gradients of all diabatic energies and then
        of the adiabatic energies requested.
        """

        en_keys = [i.replace("_grad", "") for i in self.grad_keys
                   if i.replace("_grad", "") in self.energy_keys]
        num_states = len(en_keys)
        num_mols = results[en_keys[0]].shape[0]

        # d_grads: num_diabat x num_atoms x 3
        grad_dic, d_grads, unique_diabats = self.compute_diabat_grad(
            results=results,
            xyz=xyz,
            N=N)

        # dE_dD: num_mols x num_states x num_diabat
        dE_dD = self.compute_dE_dD(results=results,
                                   en_keys=en_keys,
                                   unique_diabats=unique_diabats,
                                   num_mols=num_mols,
                                   num_states=num_states)

        # do molecule by molecule
        num_atoms = d_grads.shape[1]
        mol_d_grads = torch.split(d_grads, N, dim=1)
        all_engrads = torch.zeros(num_states, num_atoms, 3)

        counter = 0

        for i in range(num_mols):
            # num_diabat x (num_atoms of this mol) x 3
            mol_d_grad = mol_d_grads[i]

            # num_states x num_diabat
            mol_dE_dD = dE_dD[i].to(mol_d_grad.device)

            # output = num_states x (num_atoms of this_mol) x 3
            engrads = torch.einsum("ij,jkl->ikl", mol_dE_dD, mol_d_grad)

            # put into concatenated gradients
            this_num_atoms = mol_d_grad.shape[1]
            all_engrads[:, counter: counter + this_num_atoms, :] = engrads

            counter += this_num_atoms

        for j, en_key in enumerate(en_keys):
            grad_dic[en_key + "_grad"] = all_engrads[j]

        return grad_dic

    def add_grad(self,
                 results,
                 xyz,
                 N,
                 extra_grads=None,
                 try_speedup=False):

        # for example, if you want the gradients of the diabatic
        # energies

        if extra_grads is not None:

            # For two states you can get a speed-up by first
            # computing the gradients of all diabatic quantities
            # and then using the chain rule to get the adiabatic
            # gradients. This slows things down for >= 4 states
            # if you only want diagonal diabatic gradients.
            # The function `choose_grad_route` identifies which
            # method should be faster.

            # This provides a speedup on cpu but actually slows
            # things down on GPU, possibly because of having
            # to move dE_dD to the GPU. The increase in time
            # is actually only about 25% when doing in the
            # naive way for a batch size of 20 and 2 states.

            grad_route = self.choose_grad_route(extra_grads)

            if try_speedup and grad_route == "diabat_to_en":
                grads = self.compute_all_grads(results, xyz, N)
                results.update(grads)
                return results

            all_grad_keys = [*self.grad_keys, *extra_grads]
        else:
            all_grad_keys = self.grad_keys

        for grad_key in all_grad_keys:
            if "_grad" not in grad_key:
                grad_key += "_grad"

            base_key = grad_key.replace("_grad", "")
            output = results[base_key]

            grad = compute_grad(inputs=xyz, output=output)

            results[grad_key] = grad

        return results

    def add_gap(self, results):

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

    def forward(self,
                batch,
                xyz,
                results,
                add_nacv=False,
                add_grad=True,
                add_gap=True,
                extra_grads=None,
                try_speedup=False):

        if not hasattr(self, "delta"):
            self.delta = False

        if not hasattr(self, "stochastic_modules"):
            self.stochastic_modules = nn.ModuleDict({})

        if not hasattr(self, "cross_talk"):
            self.cross_talk = None

        N = batch["num_atoms"].detach().cpu().tolist()

        # preprocessing applied to diabats before computing
        # adiabats

        if self.delta:
            diag_diabat = np.array(self.diabat_keys).diagonal()
            for key in diag_diabat[1:]:
                results[key] += results[diag_diabat[0]]

        if self.cross_talk:
            results = self.cross_talk(results)

        # calculation of adiabats and their gradients

        results = self.add_diag(results=results,
                                N=N,
                                xyz=xyz,
                                add_nacv=add_nacv)

        if add_grad:
            results = self.add_grad(results=results,
                                    xyz=xyz,
                                    N=N,
                                    extra_grads=extra_grads,
                                    try_speedup=try_speedup)

        if add_gap:
            results = self.add_gap(results)

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
