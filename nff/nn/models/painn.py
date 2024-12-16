from torch import nn
import numpy as np
import copy
from nff.utils.tools import make_directed
from nff.nn.modules.painn import (
    MessageBlock,
    UpdateBlock,
    EmbeddingBlock,
    ReadoutBlock,
    ReadoutBlock_Vec,
    ReadoutBlock_Tuple,
    ReadoutBlock_Complex,
    TransformerMessageBlock,
    NbrEmbeddingBlock,
)
from nff.nn.modules.schnet import (
    AttentionPool,
    SumPool,
    MolFpPool,
    MeanPool,
    get_rij,
    add_embedding,
    add_stress,
)

from nff.nn.modules.diabat import DiabaticReadout, AdiabaticReadout
from nff.nn.layers import Diagonalize, ExpNormalBasis
from nff.utils.scatter import scatter_add, compute_grad
import torch

import pdb

POOL_DIC = {
    "sum": SumPool,
    "mean": MeanPool,
    "attention": AttentionPool,
    "mol_fp": MolFpPool,
}


class Painn(nn.Module):
    def __init__(self, modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__()

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        output_keys = modelparams["output_keys"]
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        readout_dropout = modelparams.get("readout_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")
        pool_dic = modelparams.get("pool_dic")

        self.excl_vol = modelparams.get("excl_vol", False)
        if self.excl_vol:
            self.power = modelparams["V_ex_power"]
            self.sigma = modelparams["V_ex_sigma"]

        self.grad_keys = modelparams["grad_keys"]
        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    feat_dim=feat_dim,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    learnable_k=learnable_k,
                    dropout=conv_dropout,
                )
                for _ in range(num_conv)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(
                    feat_dim=feat_dim, activation=activation, dropout=conv_dropout
                )
                for _ in range(num_conv)
            ]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = modelparams.get(
            "skip_connection", {key: False for key in self.output_keys}
        )

        self.num_readouts = num_conv if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                ReadoutBlock(
                    feat_dim=feat_dim,
                    output_keys=output_keys,
                    activation=activation,
                    dropout=readout_dropout,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(self.num_readouts)
            ]
        )

        if pool_dic is None:
            self.pool_dic = {key: SumPool() for key in self.output_keys}
        else:
            self.pool_dic = nn.ModuleDict({})
            for out_key, sub_dic in pool_dic.items():
                if out_key not in self.output_keys:
                    continue
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                self.pool_dic[out_key] = pool_class(**kwargs)

        self.compute_delta = modelparams.get("compute_delta", False)
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):
        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["features"] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out, xyz, r_ij, nbrs, inference=False):
        # import here to avoid circular imports
        from nff.train import batch_detach

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0].readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():
            grad_key = f"{key}_grad"
            grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
            if "stress" in self.grad_keys and "stress" not in all_results:
                grad_keys.append("stress")
            results = pool_obj(
                batch=batch,
                xyz=xyz,
                r_ij=r_ij,
                nbrs=nbrs,
                atomwise_output=atomwise_out,
                grad_keys=grad_keys,
                out_keys=[key],
            )

            if inference:
                results = batch_detach(results)
            all_results.update(results)

        # transfer those results that don't get pooled
        if inference:
            atomwise_out = batch_detach(atomwise_out)
        for key in atomwise_out.keys():
            if key not in all_results.keys():
                all_results[key] = atomwise_out[key]

        return all_results, xyz

    def add_delta(self, all_results):
        for i, e_i in enumerate(self.output_keys):
            if i == 0:
                continue
            e_j = self.output_keys[i - 1]
            key = f"{e_i}_{e_j}_delta"
            all_results[key] = all_results[e_i] - all_results[e_j]
            grad_keys = [e_i + "_grad", e_j + "_grad"]
            delta_grad_key = "_".join(grad_keys) + "_delta"
            if all([grad_key in all_results for grad_key in grad_keys]):
                all_results[delta_grad_key] = (
                    all_results[grad_keys[0]] - all_results[grad_keys[1]]
                )
        return all_results

    def V_ex(self, r_ij, nbr_list, xyz):
        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = (dist.reciprocal() * self.sigma).pow(self.power)

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def run(
        self,
        batch,
        xyz=None,
        requires_embedding=False,
        requires_stress=False,
        inference=False,
    ):
        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            for key in self.output_keys:
                atomwise_out[key] += r_ex

        all_results, xyz = self.pool(
            batch=batch,
            atomwise_out=atomwise_out,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            inference=inference,
        )

        if requires_embedding:
            all_results = add_embedding(
                atomwise_out=atomwise_out, all_results=all_results
            )

        if requires_stress:
            all_results = add_stress(
                batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij
            )

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        return all_results, xyz

    def forward(
        self,
        batch,
        xyz=None,
        requires_embedding=False,
        requires_stress=False,
        inference=False,
        **kwargs,
    ):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(
            batch=batch,
            xyz=xyz,
            requires_embedding=requires_embedding,
            requires_stress=requires_stress,
            inference=inference,
        )

        return results


class PainnTransformer(Painn):
    def __init__(self, modelparams):
        super().__init__(modelparams)

        conv_dropout = modelparams.get("conv_dropout", 0)
        learnable_mu = modelparams.get("learnable_mu", False)
        learnable_beta = modelparams.get("learnable_beta", False)
        same_message_blocks = modelparams["same_message_blocks"]
        feat_dim = modelparams["feat_dim"]

        rbf = ExpNormalBasis(
            n_rbf=modelparams["n_rbf"],
            cutoff=modelparams["cutoff"],
            learnable_mu=learnable_mu,
            learnable_beta=learnable_beta,
        )

        self.message_blocks = nn.ModuleList(
            [
                TransformerMessageBlock(
                    num_heads=modelparams["num_heads"],
                    feat_dim=feat_dim,
                    activation=modelparams["activation"],
                    layer_norm=modelparams.get("layer_norm", True),
                    rbf=rbf,
                )
                for _ in range(modelparams["num_conv"])
            ]
        )

        if same_message_blocks:
            self.message_blocks = nn.ModuleList(
                [self.message_blocks[0]] * len(self.message_blocks)
            )

        self.embed_block = NbrEmbeddingBlock(
            feat_dim=feat_dim, dropout=conv_dropout, rbf=rbf
        )


class PainnDiabat(Painn):
    def __init__(self, modelparams):
        """
        `diabat_keys` has the shape of a 2x2 matrix
        """

        energy_keys = modelparams["output_keys"]
        diabat_keys = modelparams["diabat_keys"]
        delta = modelparams.get("delta", False)
        new_out_keys = list(set(np.array(diabat_keys).reshape(-1).tolist()))

        new_modelparams = copy.deepcopy(modelparams)
        new_modelparams.update({"output_keys": new_out_keys, "grad_keys": []})
        super().__init__(new_modelparams)

        self.diag = Diagonalize()
        self.diabatic_readout = DiabaticReadout(
            diabat_keys=diabat_keys,
            grad_keys=modelparams["grad_keys"],
            energy_keys=energy_keys,
            delta=delta,
            stochastic_dic=modelparams.get("stochastic_dic"),
            cross_talk_dic=modelparams.get("cross_talk_dic"),
            hellmann_feynman=modelparams.get("hellmann_feynman", True),
        )
        self.add_nacv = modelparams.get("add_nacv", False)

    @property
    def _grad_keys(self):
        return self.grad_keys

    @_grad_keys.setter
    def _grad_keys(self, value):
        self.grad_keys = value
        self.diabatic_readout.grad_keys = value

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
        # for backwards compatability
        self.grad_keys = []

        if not hasattr(self, "output_keys"):
            diabat_keys = self.diabatic_readout.diabat_keys
            self.output_keys = list(set(np.array(diabat_keys).reshape(-1).tolist()))

        if hasattr(self, "add_nacv") and not inference:
            add_nacv = self.add_nacv

        diabat_results, xyz = self.run(batch=batch, xyz=xyz)
        results = self.diabatic_readout(
            batch=batch,
            xyz=xyz,
            results=diabat_results,
            add_nacv=add_nacv,
            add_grad=add_grad,
            add_gap=add_gap,
            add_u=add_u,
            inference=inference,
            do_nan=do_nan,
            en_keys_for_grad=en_keys_for_grad,
        )
        results.update({"xyz": xyz})

        return results


class PainnAdiabat(Painn):
    def __init__(self, modelparams):
        new_params = copy.deepcopy(modelparams)
        new_params.update({"grad_keys": []})

        super().__init__(new_params)

        self.adiabatic_readout = AdiabaticReadout(
            output_keys=self.output_keys,
            grad_keys=modelparams["grad_keys"],
            abs_name=modelparams["abs_fn"],
        )

    def forward(self, batch, xyz=None, add_nacv=False, add_grad=True, add_gap=True):
        results, xyz = self.run(batch=batch, xyz=xyz)
        results = self.adiabatic_readout(results=results, xyz=xyz)

        return results


class PainnGapToAbs(nn.Module):
    """
    Model for predicting a non-ground state energy, given one model that predicts
    the ground state energy and one that predicts the gap.
    """

    def __init__(self, ground_model, gap_model, subtract_gap):
        super(PainnGapToAbs, self).__init__()

        self.ground_model = ground_model
        self.gap_model = gap_model
        self.subtract_gap = subtract_gap
        self.models = [self.ground_model, self.gap_model]

    def get_model_attr(self, model, key):
        if hasattr(model, "painn_model"):
            return getattr(model.painn_model, key)
        return getattr(model, key)

    def set_model_attr(self, model, key, val):
        if hasattr(model, "painn_model"):
            sub_model = model.painn_model
        else:
            sub_model = model

        setattr(sub_model, key, val)

    def get_grad_keys(self, model):
        if hasattr(model, "painn_model"):
            grad_keys = model.painn_model.grad_keys
        else:
            grad_keys = model.grad_keys
        return set(grad_keys)

    @property
    def grad_keys(self):
        ground_grads = set(self.get_model_attr(self.ground_model, "grad_keys"))
        gap_grads = set(self.get_model_attr(self.gap_model, "grad_keys"))
        common_grads = [i for i in ground_grads if i in gap_grads]

        return common_grads

    @grad_keys.setter
    def grad_keys(self, value):
        self.set_model_attr(self.ground_model, "grad_keys", value)
        self.set_model_attr(self.gap_model, "grad_keys", value)

    def forward(self, *args, **kwargs):
        ground_results = self.ground_model(*args, **kwargs)
        gap_results = self.gap_model(*args, **kwargs)

        common_keys = list(set(list(ground_results.keys()) + list(gap_results.keys())))

        factor = -1 if self.subtract_gap else 1
        combined_results = {}

        for key in common_keys:
            pool_dics = [
                self.get_model_attr(model, "pool_dic") for model in self.models
            ]

            in_pool = all([key in dic for dic in pool_dics])
            in_grad = all(
                [
                    key in self.get_model_attr(model, "grad_keys")
                    for model in self.models
                ]
            )

            common = in_pool or in_grad

            if not common:
                continue

            val = ground_results[key] + factor * gap_results[key]
            combined_results[key] = val

        return combined_results


class Painn_VecOut(Painn):
    def __init__(self, modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__(modelparams)

        output_vec_keys = modelparams["output_vec_keys"]
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        readout_dropout = modelparams.get("readout_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")

        self.output_vec_keys = output_vec_keys
        # no skip connection in original paper
        self.skip_vec = modelparams.get(
            "skip_vec_connection", {key: False for key in self.output_vec_keys}
        )

        num_vec_readouts = modelparams["num_conv"] if any(self.skip.values()) else 1
        self.readout_vec_blocks = nn.ModuleList(
            [
                ReadoutBlock_Vec(
                    feat_dim=feat_dim,
                    output_keys=output_vec_keys,
                    activation=activation,
                    dropout=readout_dropout,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(num_vec_readouts)
            ]
        )

    def atomwise(self, batch, xyz=None):
        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            readout_vec_block = self.readout_vec_blocks[i]
            new_vec_results = readout_vec_block(s_i=s_i, v_i=v_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

            first_vec_readout = self.readout_vec_blocks[0]
            new_vec_results = first_vec_readout(s_i=s_i, v_i=v_i)
            for key, skip in self.skip_vec.items():
                if key not in new_vec_results:
                    continue
                if not skip:
                    results[key] = new_vec_results[key]

        results["features"] = s_i
        results["features_vec"] = v_i

        return results, xyz, r_ij, nbrs


class Painn_VecOut2(Painn_VecOut):
    # unlike Painn_VecOut this uses 2 equivariant blocks for each output
    def __init__(self,
                 modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        super().__init__(modelparams)

        output_vec_keys = modelparams["output_vec_keys"]
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        readout_dropout = modelparams.get("readout_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")

        self.output_vec_keys = output_vec_keys
        # no skip connection in original paper
        self.skip_vec = modelparams.get("skip_vec_connection",
                                        {key: False for key
                                         in self.output_vec_keys})

        num_vec_readouts = (modelparams["num_conv"] if any(self.skip.values())
                            else 1)
        self.readout_vec_blocks = nn.ModuleList(
            [ReadoutBlock_Vec2(feat_dim=feat_dim,
                               output_keys=output_vec_keys,
                               activation=activation,
                               dropout=readout_dropout,
                               means=means,
                               stddevs=stddevs)
             for _ in range(num_vec_readouts)]
        )


class Painn_NAC_OuterProd(Painn_VecOut2):
    # This model attempts to learn non-adiabatic coupling vectors
    # as suggested by Jeremy Richardson, as eigenvector
    # of an outer product matrix
    def __init__(self,
                 modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        super().__init__(modelparams)

    def get_nac(self,
                all_results,
                batch,
                xyz):

        N = batch["num_atoms"].detach().cpu().tolist()
        xyz_s = torch.split(xyz, N)

        for key in self.output_vec_keys:

            mats = []
            nacs = []
            nu_s = torch.split(all_results[key], N)
            for nu, r in zip(nu_s, xyz_s):
                mat = (torch.outer(r.reshape(-1), nu.reshape(-1))
                       + torch.outer(nu.reshape(-1), r.reshape(-1)))
                mats.append(mat)

                eigvals, eigvecs = torch.linalg.eigh(mat)
                real_vals = torch.abs(eigvals)
                phase = eigvals[0] / real_vals[0]
                real_vecs = torch.real(eigvecs / phase)

                max_idx = torch.argmax(real_vals)
                nac = real_vecs[:, max_idx] * torch.sqrt(real_vals[max_idx])
                nacs.append(nac.reshape(-1, 3))

            all_results[key] = torch.cat(nacs)
            all_results[key + "_mat"] = tuple(mats)

        return all_results, xyz

    def run(self,
            batch,
            xyz=None,
            requires_embedding=False,
            requires_stress=False,
            inference=False):

        from nff.train import batch_detach

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch,
                                                      xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            for key in self.output_keys:
                atomwise_out[key] += r_ex

        pooled_results, xyz = self.pool(batch=batch,
                                        atomwise_out=atomwise_out,
                                        xyz=xyz,
                                        r_ij=r_ij,
                                        nbrs=nbrs,
                                        inference=False)

        all_results, xyz = self.get_nac(all_results=pooled_results,
                                        batch=batch,
                                        xyz=xyz)

        if requires_embedding:
            all_results = add_embedding(atomwise_out=atomwise_out,
                                        all_results=all_results)

        if requires_stress:
            all_results = add_stress(batch=batch,
                                     all_results=all_results,
                                     nbrs=nbrs,
                                     r_ij=r_ij)

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        if inference:
            batch_detach(all_results)

        return all_results, xyz


class Painn_Complex(Painn):

    def __init__(self,
                 modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__(modelparams)

        self.output_cmplx_keys = modelparams["output_cmplx_keys"]
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        readout_dropout = modelparams.get("readout_dropout", 0)

        num_cmplx_readouts = (modelparams["num_conv"] if any(self.skip.values())
                              else 1)
        self.readout_cmplx_blocks = nn.ModuleList(
            [ReadoutBlock_Complex(feat_dim=feat_dim,
                                  output_keys=self.output_cmplx_keys,
                                  activation=activation,
                                  dropout=readout_dropout)
             for _ in range(num_cmplx_readouts)]
        )

    def atomwise(self,
                 batch,
                 xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip
                         for key in self.output_keys}

        nbrs, _ = make_directed(batch['nbr_list'])
        nxyz = batch['nxyz']

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz,
                             batch=batch,
                             nbrs=nbrs,
                             cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers,
                                    nbrs=nbrs,
                                    r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs)

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            readout_cmplx_block = self.readout_cmplx_blocks[i]
            new_cmplx_results = readout_cmplx_block(s_i=s_i, v_i=v_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

            first_cmplx_readout = self.readout_cmplx_blocks[0]
            new_cmplx_results = first_cmplx_readout(s_i=s_i, v_i=v_i)
            for key, skip in self.skip_cmplx.items():
                if key not in new_cmplx_results:
                    continue
                if not skip:
                    results[key] = new_cmplx_results[key]

        results['features'] = s_i
        results['features_vec'] = v_i

        return results, xyz, r_ij, nbrs


class Painn_Tuple(Painn):
    def __init__(self, modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__(modelparams)

        self.output_tuple_keys = modelparams["output_tuple_keys"]

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        readout_dropout = modelparams.get("readout_dropout", 0)

        # no skip connection in original paper
        self.skip_tuple = modelparams.get(
            "skip_tuple_connection", {key: False for key in self.output_tuple_keys}
        )

        num_tuple_readouts = (
            modelparams["num_conv"] if any(self.skip_tuple.values()) else 1
        )
        self.readout_tuple_blocks = nn.ModuleList(
            [
                ReadoutBlock_Tuple(
                    feat_dim=feat_dim,
                    output_keys=self.output_tuple_keys,
                    activation=activation,
                    dropout=readout_dropout,
                )
                for _ in range(num_tuple_readouts)
            ]
        )

        for keys in self.output_tuple_keys:
            for key in keys.split("+"):
                self.pool_dic[key] = SumPool()

    def atomwise(self, batch, xyz=None):
        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            readout_tuple_block = self.readout_tuple_blocks[i]
            new_tuple_results = readout_tuple_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

            first_tuple_readout = self.readout_tuple_blocks[0]
            new_tuple_results = first_tuple_readout(s_i=s_i)
            for keys, skip in self.skip_tuple.items():
                for key in keys.split("+"):  # bc they are a tuple
                    if key not in new_tuple_results:
                        continue
                    if not skip:
                        results[key] = new_tuple_results[key]

        results["features"] = s_i
        results["features_vec"] = v_i

        return results, xyz, r_ij, nbrs


class Painn_wCP(Painn_Tuple):
    def __init__(self, modelparams):
        """
        This model tries to implement the Characteristic
        Polynomial Approach from:

        Tzu Yu Wang, Simon P. Neville, and Michael S. Schuurman
        J. Phys. Chem. Lett. 2023, 14, 7780−7786

        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__(modelparams)

        self.real_output_tuple_keys = copy.deepcopy(self.output_tuple_keys)

        self.output_tuple_keys = []
        for ii, longkey in enumerate(self.real_output_tuple_keys):
            num_energies = len(longkey.split("+"))
            string = f"omega_{ii}"
            for cc in range(num_energies - 1):
                string += f"+c{cc:02d}_{ii}"
            self.output_tuple_keys.append(string)

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        readout_dropout = modelparams.get("readout_dropout", 0)

        # no skip connection in original paper
        self.skip_tuple = modelparams.get(
            "skip_tuple_connection", {key: False for key in self.output_tuple_keys}
        )

        num_tuple_readouts = (
            modelparams["num_conv"] if any(self.skip_tuple.values()) else 1
        )
        # overwrite what has been done before
        self.readout_tuple_blocks = nn.ModuleList(
            [
                ReadoutBlock_Tuple(
                    feat_dim=feat_dim,
                    output_keys=self.output_tuple_keys,
                    activation=activation,
                    dropout=readout_dropout,
                )
                for _ in range(num_tuple_readouts)
            ]
        )

        for keys in self.output_tuple_keys:
            for key in keys.split("+"):
                self.pool_dic[key] = SumPool()

        # remove the energy keys that were added
        for keys in self.real_output_tuple_keys:
            for key in keys.split("+"):
                _ = self.pool_dic.pop(key)

    def adibatic_energies(
        self,
        all_results,
        xyz=None,
    ):
        for long1, long2 in zip(self.output_tuple_keys, self.real_output_tuple_keys):
            wCP_keys = long1.split("+")
            adiabat_keys = long2.split("+")

            num_states = len(wCP_keys)
            omega = all_results[wCP_keys[0]]
            batch_size = len(omega)

            C_mat0 = torch.zeros(
                batch_size, num_states, num_states, device=omega.device
            )

            for mat in C_mat0:
                mat.fill_diagonal_(1)
            C_mat = C_mat0 * omega.reshape(-1, 1, 1) + torch.diag(
                torch.ones(num_states - 1), -1
            ).to(omega.device)

            for idx, coef in enumerate(wCP_keys[1:]):
                C_mat[:, idx, -1] = -all_results[coef]

            all_results[f"Cmat_{wCP_keys[0][-1]}"] = C_mat

            # eigvals = torch.real(torch.linalg.eigvals(C_mat))
            eigvals = torch.abs(torch.linalg.eigvals(C_mat))
            for column, key in zip(eigvals.T, adiabat_keys[::-1]):
                all_results[key] = column

            for key in adiabat_keys:
                grad_key = f"{key}_grad"
                if grad_key in self.grad_keys:
                    output = all_results[key]
                    grad = compute_grad(output=output, inputs=xyz)
                    all_results[grad_key] = grad

        return all_results, xyz

    def run(
        self,
        batch,
        xyz=None,
        requires_embedding=False,
        requires_stress=False,
        inference=False,
    ):
        from nff.train import batch_detach

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            for key in self.output_keys:
                atomwise_out[key] += r_ex

        intermediate_results, xyz = self.pool(
            batch=batch,
            atomwise_out=atomwise_out,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            inference=False,
        )

        all_results, xyz = self.adibatic_energies(
            all_results=intermediate_results, xyz=xyz
        )

        if requires_embedding:
            all_results = add_embedding(
                atomwise_out=atomwise_out, all_results=all_results
            )

        if requires_stress:
            all_results = add_stress(
                batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij
            )

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        if inference:
            batch_detach(all_results)

        return all_results, xyz


# same as Painn_wCP but not based on the tuple architecture
# seems to perform worse, thus commented out for now,
# maybe it should be delted
# class Painn_wCP2(Painn):

#     def __init__(self,
#                  modelparams):
#         """
#         This model tries to implement the Characteristic
#         Polynomial Approach from:

#         Tzu Yu Wang, Simon P. Neville, and Michael S. Schuurman
#         J. Phys. Chem. Lett. 2023, 14, 7780−7786

#         Args:
#             modelparams (dict): dictionary of model parameters


#         """

#         super().__init__(modelparams)

#         self.output_tuple_keys = modelparams["output_tuple_keys"]
#         #number_wCP_layers = len(output_tuple_keys)
#         self.wCP_keys = [[]]

#         for longkey in self.output_tuple_keys:
#             keys = longkey.split('+')
#             for key in keys:
#                 _ = self.pool_dic.pop(key)
#                 self.output_keys.remove(key)

#         for ii, longkey in enumerate(self.output_tuple_keys):
#             num_energies = len(longkey.split('+'))
#             self.wCP_keys[-1].append(f"omega_{ii}")
#             self.output_keys.append(f"omega_{ii}")
#             for cc in range(num_energies -1):
#                 self.wCP_keys[-1].append(f"c{cc:02d}_{ii}")
#                 self.output_keys.append(f"c{cc:02d}_{ii}")

#         feat_dim = modelparams["feat_dim"]
#         activation = modelparams["activation"]
#         readout_dropout = modelparams.get("readout_dropout", 0)
#         means = modelparams.get("means")
#         stddevs = modelparams.get("stddevs")

#         for key in self.output_keys:
#             if key not in self.skip.keys():
#                 self.skip.update({key: False})

#         self.readout_blocks = nn.ModuleList(
#             [ReadoutBlock(feat_dim=feat_dim,
#                           output_keys=self.output_keys,
#                           activation=activation,
#                           dropout=readout_dropout,
#                           means=means,
#                           stddevs=stddevs)
#              for _ in range(self.num_readouts)]
#         )

#         for keys in self.output_keys:
#             for key in keys.split("+"):
#                 self.pool_dic[key] = SumPool()

#     def adiabatic_energies(self,
#                           all_results,
#                           xyz=None,):

#         for list1, long2 in zip(self.wCP_keys,
#                                 self.output_tuple_keys):

#             wCP_keys     = list1
#             adiabat_keys = long2.split('+')

#             num_states = len(wCP_keys)
#             omega = all_results[wCP_keys[0]]
#             batch_size = len(omega)

#             C_mat0 = torch.zeros(batch_size, num_states, num_states,
#                                  device=omega.device)
#             #pdb.set_trace()

#             for mat in C_mat0:
#                 mat.fill_diagonal_(1)
#             C_mat = (C_mat0 * omega.reshape(-1, 1, 1)
#                                + torch.diag(torch.ones(num_states-1), -1).to(omega.device)
#                     )

#             for idx, coef in enumerate(wCP_keys[1:]):
#                 C_mat[:, idx, -1] = - all_results[coef]

#             all_results[f"Cmat_{wCP_keys[0][-1]}"] = C_mat

#             #eigvals = torch.real(torch.linalg.eigvals(C_mat))
#             eigvals = torch.abs(torch.linalg.eigvals(C_mat))
#             for column, key in zip(eigvals.T, adiabat_keys[::-1]):
#                 all_results[key] = column

#             for key in adiabat_keys:
#                 grad_key = f"{key}_grad"
#                 if grad_key in self.grad_keys:
#                     output   = all_results[key]
#                     grad     = compute_grad(output=output,
#                                             inputs=xyz)
#                     all_results[grad_key] = grad

#     def run(self,
#             batch,
#             xyz=None,
#             requires_embedding=False,
#             requires_stress=False,
#             inference=False):

#         from nff.train import batch_detach

#         atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch,
#                                                       xyz=xyz)

#         if getattr(self, "excl_vol", None):
#             # Excluded Volume interactions
#             r_ex = self.V_ex(r_ij, nbrs, xyz)
#             for key in self.output_keys:
#                 atomwise_out[key] += r_ex

#         all_results, xyz = self.pool(batch=batch,
#                                      atomwise_out=atomwise_out,
#                                      xyz=xyz,
#                                      r_ij=r_ij,
#                                      nbrs=nbrs,
#                                      inference=False)

#         self.adiabatic_energies(all_results=all_results,
#                                xyz=xyz)

#         if requires_embedding:
#             all_results = add_embedding(atomwise_out=atomwise_out,
#                                         all_results=all_results)

#         if requires_stress:
#             all_results = add_stress(batch=batch,
#                                      all_results=all_results,
#                                      nbrs=nbrs,
#                                      r_ij=r_ij)

#         if getattr(self, "compute_delta", False):
#             all_results = self.add_delta(all_results)

#         if inference:
#             batch_detach(all_results)

#         return all_results, xyz


class PainnDipole(Painn_VecOut):
    """
    Model class that basically does the same thing as Painn_VecOut,
    but is slightly modified for Simon's applications of dipole
    moments. Made into a separate class to not interfere with
    Johannes' work.
    """

    def __init__(self, modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__(modelparams)

        # dictionary of the form {key: True/False} for all keys
        # that are vector outputs. If True, then the output is
        # a vector for each atom. If False, then the per-atom
        # vectors are summed to give a vector per molecule (like
        # a dipole moment)
        self.vector_per_atom = modelparams["vector_per_atom"]

    def forward(
        self,
        batch,
        xyz=None,
        requires_embedding=False,
        requires_stress=False,
        inference=False,
        **kwargs,
    ):
        results = super().forward(
            batch=batch,
            xyz=xyz,
            requires_embedding=requires_embedding,
            requires_stress=requires_stress,
            inference=inference,
            **kwargs,
        )

        # sum the per-atom vectors for each molecule, if necessary
        for key in self.output_vec_keys:
            if self.vector_per_atom[key]:
                continue
            val = results[key]
            split_vals = torch.split(val, batch["num_atoms"].tolist())
            final_vals = torch.stack(
                [split_val.sum(0).reshape(3) for split_val in split_vals]
            )
            results[key] = final_vals

        return results
