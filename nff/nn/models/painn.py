import torch
from torch import nn
import numpy as np
import copy

from nff.utils.tools import make_directed
from nff.nn.modules.painn import (MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock)
from nff.nn.modules.schnet import (AttentionPool, SumPool)
#from nff.nn.modules.diabat import DiabaticReadout
#from nff.nn.layers import Diagonalize


POOL_DIC = {"sum": SumPool,
            "attention": AttentionPool}


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

        num_readouts = num_conv if (self.skip) else 1
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
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                self.pool_dic[out_key] = pool_class(**kwargs)

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
        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]

        s_i, v_i = self.embed_block(z_numbers)
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
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    results[key] = new_results[key]

        results['features'] = s_i

        return results, xyz

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

    def run(self,
            batch,
            xyz=None):

        atomwise_out, xyz = self.atomwise(batch=batch,
                                          xyz=xyz)
        all_results, xyz = self.pool(batch=batch,
                                     atomwise_out=atomwise_out,
                                     xyz=xyz)

        return all_results, xyz

    def forward(self, batch, xyz=None):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(batch=batch,
                              xyz=xyz)
        return results

