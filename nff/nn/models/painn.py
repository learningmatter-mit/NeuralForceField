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
             xyz):

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

    def run(self,
            batch,
            xyz=None,
            requires_stress=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch,
                                                      xyz=xyz)
        all_results, xyz = self.pool(batch=batch,
                                     atomwise_out=atomwise_out,
                                     xyz=xyz)

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
                xyz=None, requires_stress=False):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(batch=batch,
                              xyz=xyz, requires_stress=requires_stress)

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
                add_nacv=False,
                add_grad=True,
                add_gap=True,
                add_u=False):

        # for backwards compatability
        self.grad_keys = []

        if not hasattr(self, "output_keys"):
            diabat_keys = self.diabatic_readout.diabat_keys
            self.output_keys = list(set(np.array(diabat_keys)
                                        .reshape(-1)
                                        .tolist()))
        if hasattr(self, "add_nacv"):
            add_nacv = self.add_nacv

        diabat_results, xyz = self.run(batch=batch,
                                       xyz=xyz)
        results = self.diabatic_readout(batch=batch,
                                        xyz=xyz,
                                        results=diabat_results,
                                        add_nacv=add_nacv,
                                        add_grad=add_grad,
                                        add_gap=add_gap,
                                        add_u=add_u)

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
