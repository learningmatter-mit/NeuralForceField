from torch import nn
import numpy as np
import copy
from nff.utils.tools import make_directed
from nff.nn.modules.painn import (MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock,
                                  TransformerMessageBlock,
                                  NbrEmbeddingBlock)
from nff.nn.modules.schnet import (AttentionPool, SumPool, MolFpPool,
                                   MeanPool, get_rij, add_stress)
from nff.nn.modules.diabat import DiabaticReadout, AdiabaticReadout
from nff.nn.layers import (Diagonalize, ExpNormalBasis)
from nff.utils.scatter import scatter_add

POOL_DIC = {"sum": SumPool,
            "mean": MeanPool,
            "attention": AttentionPool,
            "mol_fp": MolFpPool}


class Painn(nn.Module):

    def __init__(self,
                 modelparams):
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
            [MessageBlock(feat_dim=feat_dim,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=learnable_k,
                          dropout=conv_dropout)
             for _ in range(num_conv)]
        )
        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=feat_dim,
                         activation=activation,
                         dropout=conv_dropout)
             for _ in range(num_conv)]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = modelparams.get("skip_connection",
                                    {key: False for key
                                     in self.output_keys})

        num_readouts = num_conv if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [ReadoutBlock(feat_dim=feat_dim,
                          output_keys=output_keys,
                          activation=activation,
                          dropout=readout_dropout,
                          means=means,
                          stddevs=stddevs)
             for _ in range(num_readouts)]
        )

        if pool_dic is None:
            self.pool_dic = {key: SumPool() for key
                             in self.output_keys}
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

        results['features'] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self,
             batch,
             atomwise_out,
             xyz,
             inference=False):

        # import here to avoid circular imports
        from nff.train import batch_detach

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0]
                                    .readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key
                             in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():

            grad_key = f"{key}_grad"
            grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
            results = pool_obj(batch=batch,
                               xyz=xyz,
                               atomwise_output=atomwise_out,
                               grad_keys=grad_keys,
                               out_keys=[key])

            if inference:
                results = batch_detach(results)
            all_results.update(results)

        return all_results, xyz

    def add_delta(self, all_results):
        for i, e_i in enumerate(self.output_keys):
            if i == 0:
                continue
            e_j = self.output_keys[i-1]
            key = f"{e_i}_{e_j}_delta"
            all_results[key] = (all_results[e_i] -
                                all_results[e_j])
        return all_results

    def V_ex(self, r_ij, nbr_list, xyz):

        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = ((dist.reciprocal() * self.sigma).pow(self.power))

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def run(self,
            batch,
            xyz=None,
            requires_stress=False,
            inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch,
                                                      xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            atomwise_out['energy'] += r_ex

        all_results, xyz = self.pool(batch=batch,
                                     atomwise_out=atomwise_out,
                                     xyz=xyz,
                                     inference=inference)

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)
        if requires_stress:
            all_results = add_stress(batch=batch,
                                     all_results=all_results,
                                     nbrs=nbrs,
                                     r_ij=r_ij)
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


class PainnTransformer(Painn):
    def __init__(self,
                 modelparams):
        super().__init__(modelparams)

        conv_dropout = modelparams.get("conv_dropout", 0)
        learnable_mu = modelparams.get("learnable_mu", False)
        learnable_beta = modelparams.get("learnable_beta", False)
        same_message_blocks = modelparams["same_message_blocks"]
        feat_dim = modelparams["feat_dim"]

        rbf = ExpNormalBasis(n_rbf=modelparams["n_rbf"],
                             cutoff=modelparams["cutoff"],
                             learnable_mu=learnable_mu,
                             learnable_beta=learnable_beta)

        self.message_blocks = nn.ModuleList(
            [
                TransformerMessageBlock(
                    num_heads=modelparams["num_heads"],
                    feat_dim=feat_dim,
                    activation=modelparams["activation"],
                    layer_norm=modelparams.get("layer_norm", True),
                    rbf=rbf)

                for _ in range(modelparams["num_conv"])
            ]
        )

        if same_message_blocks:
            self.message_blocks = nn.ModuleList(
                [self.message_blocks[0]]
                * len(self.message_blocks))

        self.embed_block = NbrEmbeddingBlock(feat_dim=feat_dim,
                                             dropout=conv_dropout,
                                             rbf=rbf)


class PainnDiabat(Painn):

    def __init__(self, modelparams):
        """
        `diabat_keys` has the shape of a 2x2 matrix
        """

        energy_keys = modelparams["output_keys"]
        diabat_keys = modelparams["diabat_keys"]
        delta = modelparams.get("delta", False)
        new_out_keys = list(set(np.array(diabat_keys).reshape(-1)
                                .tolist()))

        new_modelparams = copy.deepcopy(modelparams)
        new_modelparams.update({"output_keys": new_out_keys,
                                "grad_keys": []})
        super().__init__(new_modelparams)

        self.diag = Diagonalize()
        self.diabatic_readout = DiabaticReadout(
            diabat_keys=diabat_keys,
            grad_keys=modelparams["grad_keys"],
            energy_keys=energy_keys,
            delta=delta,
            stochastic_dic=modelparams.get("stochastic_dic"),
            cross_talk_dic=modelparams.get("cross_talk_dic"),
            hellmann_feynman=modelparams.get("hellmann_feynman", True))
        self.add_nacv = modelparams.get("add_nacv", False)

    @property
    def _grad_keys(self):
        return self.grad_keys

    @_grad_keys.setter
    def _grad_keys(self, value):
        self.grad_keys = value
        self.diabatic_readout.grad_keys = value

    def forward(self,
                batch,
                xyz=None,
                add_nacv=True,
                add_grad=True,
                add_gap=True,
                add_u=False,
                inference=False,
                do_nan=True,
                en_keys_for_grad=None):

        # for backwards compatability
        self.grad_keys = []

        if not hasattr(self, "output_keys"):
            diabat_keys = self.diabatic_readout.diabat_keys
            self.output_keys = list(set(np.array(diabat_keys)
                                        .reshape(-1)
                                        .tolist()))

        if hasattr(self, "add_nacv") and not inference:
            add_nacv = self.add_nacv

        diabat_results, xyz = self.run(batch=batch,
                                       xyz=xyz)
        results = self.diabatic_readout(batch=batch,
                                        xyz=xyz,
                                        results=diabat_results,
                                        add_nacv=add_nacv,
                                        add_grad=add_grad,
                                        add_gap=add_gap,
                                        add_u=add_u,
                                        inference=inference,
                                        do_nan=do_nan,
                                        en_keys_for_grad=en_keys_for_grad)

        return results


class PainnAdiabat(Painn):

    def __init__(self, modelparams):
        new_params = copy.deepcopy(modelparams)
        new_params.update({"grad_keys": []})

        super().__init__(new_params)

        self.adiabatic_readout = AdiabaticReadout(
            output_keys=self.output_keys,
            grad_keys=modelparams["grad_keys"],
            abs_name=modelparams["abs_fn"])

    def forward(self,
                batch,
                xyz=None,
                add_nacv=False,
                add_grad=True,
                add_gap=True):

        results, xyz = self.run(batch=batch,
                                xyz=xyz)
        results = self.adiabatic_readout(results=results,
                                         xyz=xyz)

        return results


class PainnGapToAbs(nn.Module):
    """
    Model for predicting a non-ground state energy, given one model that predicts
    the ground state energy and one that predicts the gap.
    """

    def __init__(self,
                 ground_model,
                 gap_model,
                 subtract_gap):

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
        ground_grads = set(self.get_model_attr(self.ground_model, 'grad_keys'))
        gap_grads = set(self.get_model_attr(self.gap_model, 'grad_keys'))
        common_grads = [i for i in ground_grads if i in
                        gap_grads]

        return common_grads

    @grad_keys.setter
    def grad_keys(self, value):
        self.set_model_attr(self.ground_model, 'grad_keys', value)
        self.set_model_attr(self.gap_model, 'grad_keys', value)

    def forward(self,
                *args,
                **kwargs):

        ground_results = self.ground_model(*args, **kwargs)
        gap_results = self.gap_model(*args, **kwargs)

        common_keys = list(set(list(ground_results.keys()) +
                               list(gap_results.keys())))

        factor = -1 if self.subtract_gap else 1
        combined_results = {}

        for key in common_keys:
            pool_dics = [self.get_model_attr(model, 'pool_dic') for
                         model in self.models]

            in_pool = all([key in dic for dic in pool_dics])
            in_grad = all([key in self.get_model_attr(model, 'grad_keys') for
                           model in self.models])

            common = in_pool or in_grad

            if not common:
                continue

            val = ground_results[key] + factor * gap_results[key]
            combined_results[key] = val

        return combined_results
