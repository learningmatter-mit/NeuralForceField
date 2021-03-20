import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy

from nff.utils.tools import make_directed
from nff.nn.modules.painn import (MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock)
from nff.nn.modules.schnet import (AttentionPool, SumPool)
# from nff.nn.modules.diabat import DiabaticReadout
# from nff.nn.layers import Diagonalize
from nff.nn.layers import GaussianSmearing, Dense
from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types


# POOL_DIC = {"sum": SumPool,
#             "attention": AttentionPool}

def to_module(activation):
    return layer_types[activation]()

class EdgeReadoutBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout):
        super().__init__()

        self.edgereadout = Sequential(
            Dense(in_features=feat_dim, 
                  out_features=feat_dim//2,
                  bias=True,
                  dropout_rate=dropout,
                  activation=to_module(activation)),
            Dense(in_features=feat_dim//2, 
                  out_features=1,
                  bias=True,
                  dropout_rate=dropout)
            )

    def forward(self, e):
        
        return self.edgereadout(e)



class ForcePai(nn.Module):
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

        # embedding layers
        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        # distance transform
        # cut_off = modelparams.get("cut_off")
        # n_gaussians = modelparams.get("n_gaussians")
        # n_edge_basis = modelparams.get("n_edge_basis")
        # self.smear = GaussianSmearing(start=0.0, stop=cutoff, n_gaussians=n_gaussians)
        # self.edgefilter = Sequential(
        #     Dense(in_features=n_gaussians, out_features=n_edge_basis),
        #     shifted_softplus(),
        #     Dense(in_features=n_edge_basis, out_features=n_edge_basis))
        
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

        self.edgereadout = EdgeReadoutBlock(feat_dim=feat_dim,
                                            activation=activation,
                                            dropout=readout_dropout)
        # self.readout_blocks = nn.ModuleList(
        #     [ReadoutBlock(feat_dim=feat_dim,
        #                   output_keys=output_keys,
        #                   activation=activation,
        #                   dropout=readout_dropout,
        #                   means=means,
        #                   stddevs=stddevs)
        #      for _ in range(num_readouts)]
        # )

        # if pool_dic is None:
        #     self.pool_dic = {key: SumPool() for key
        #                      in self.output_keys}
        # else:
        #     self.pool_dic = nn.ModuleDict({})
        #     for out_key, sub_dic in pool_dic.items():
        #         pool_name = sub_dic["name"].lower()
        #         kwargs = sub_dic["param"]
        #         pool_class = POOL_DIC[pool_name]
        #         self.pool_dic[out_key] = pool_class(**kwargs)

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

        # node features
        z_numbers = nxyz[:, 0].long()
        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]

        s_i, v_i = self.embed_block(z_numbers)

        # graph size
        graph_size = z_numbers.shape[0]

        # edge features
        dis_vec = xyz[nbrs[:, 0]] - xyz[nbrs[:, 1]]
        dis = dis_vec.pow(2).sum(1).sqrt()[:, None]

        xyz_adjoint = dis_vec / dis

        # e = self.smear(dis)
        # e = self.edgefilter(e)

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

        e = s_i[nbrs[:, 0]] + s_i[nbrs[:, 1]]
        f_edge = self.edgereadout(e) * xyz_adjoint
        f_atom = scatter_add(f_edge, nbrs[:,0], dim=0, dim_size=graph_size) - \
            scatter_add(f_edge, nbrs[:,1], dim=0, dim_size=graph_size)

        results['energy_grad'] = f_atom

        return results, xyz

    def run(self,
            batch,
            xyz=None):

        all_results, xyz = self.atomwise(batch=batch,
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

