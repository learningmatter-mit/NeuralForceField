import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy

from nff.utils.tools import make_directed
from nff.nn.modules.painn import (MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock)
from nff.nn.layers import PainnRadialBasis, CosineEnvelope
from nff.nn.modules.schnet import (AttentionPool, SumPool)
# from nff.nn.modules.diabat import DiabaticReadout
# from nff.nn.layers import Diagonalize
from nff.nn.layers import GaussianSmearing, Dense
from nff.nn.activations import shifted_softplus
from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types


# POOL_DIC = {"sum": SumPool,
#             "attention": AttentionPool}

EPS = 1e-15


def to_module(activation):
    return layer_types[activation]()

def norm(vec):
    result = ((vec ** 2 + EPS).sum(-1)) ** 0.5
    return result

def preprocess_r(r_ij):
    """
    r_ij (n_nbrs x 3): tensor of interatomic vectors (r_j - r_i)
    """

    dist = norm(r_ij)
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit

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
class InvariantDense(nn.Module):
    def __init__(self,
                 dim,
                 dropout,
                 activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(Dense(in_features=dim,
                                          out_features=dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=dim,
                                          out_features=3 * dim,
                                          bias=True,
                                          dropout_rate=dropout))

    def forward(self, s_j):
        output = self.layers(s_j)
        return output


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 feat_dim,
                 learnable_k,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_rbf=n_rbf,
                               cutoff=cutoff,
                               learnable_k=learnable_k)
        dense = Dense(in_features=n_rbf,
                      out_features=3 * feat_dim,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()

        self.inv_dense = InvariantDense(dim=feat_dim,
                                        activation=activation,
                                        dropout=dropout)
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=learnable_k,
                                        dropout=dropout)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x feat_dim
        out_reshape = output.reshape(output.shape[0], 3, -1)

        return out_reshape


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()
        self.inv_message = InvariantMessage(feat_dim=feat_dim,
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            learnable_k=learnable_k,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        # graph_size = s_j.shape[0]
        # delta_v_i = scatter_add(src=delta_v_ij,
        #                         index=nbrs[:, 0],
        #                         dim=0,
        #                         dim_size=graph_size)

        # delta_s_i = scatter_add(src=delta_s_ij,
        #                         index=nbrs[:, 0],
        #                         dim=0,
        #                         dim_size=graph_size)

        return delta_s_ij, delta_v_ij


class ForcePai(nn.Module):
    def __init__(self,
                 modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters
        """

        super().__init__()

        feat_dim = modelparams["feat_dim"]
        activation = modelparams.get("activation", "swish")
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        output_keys = modelparams.get("output_keys", ["energy"])
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        readout_dropout = modelparams.get("readout_dropout", 0)
        # means = modelparams.get("means")
        # stddevs = modelparams.get("stddevs")
        pool_dic = modelparams.get("pool_dic")

        self.grad_keys = modelparams.get("grad_keys", ["energy_grad"])

        # embedding layers
        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        # distance transform
        self.smear = GaussianSmearing(start=0.0, stop=cutoff, n_gaussians=n_rbf)
        self.edgefilter = Sequential(
            Dense(in_features=n_rbf, out_features=2*feat_dim, bias=True),
            shifted_softplus(),
            Dense(in_features=2*feat_dim, out_features=feat_dim))
        
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

        # construct edge update networks
        # edge_update = [Dense(in_features=feat_dim, out_features=2*feat_dim), shifted_softplus()]
        # for i in range(edge_update_depth):
        #     edge_update.append(Dense(in_features=feat_dim, out_features=feat_dim))
        #     edge_update.append(shifted_softplus())
        # edge_update.append(Dense(in_features=2*feat_dim, out_features=feat_dim))
        self.edge_update = nn.ModuleList(
            [nn.Sequential(Dense(in_features=feat_dim, 
                                out_features=2*feat_dim), 
                          shifted_softplus(),
                          Dense(in_features=2*feat_dim, 
                                out_features=feat_dim))
            for _ in range(num_conv)]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = modelparams.get("skip_connection",
                                    {key: False for key
                                     in self.output_keys})

        num_readouts = num_conv if (self.skip) else 1

        # edge readout 
        self.edgereadout = Sequential(
            Dense(in_features=feat_dim, 
                    out_features=feat_dim, 
                    bias=True, 
                    dropout_rate=readout_dropout),
            shifted_softplus(),
            Dense(in_features=feat_dim, 
                    out_features=1, 
                    dropout_rate=readout_dropout)
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
        e_ij = self.smear(dis)
        e_ij = self.edgefilter(e_ij)

        results = {}

        for i, message_block in enumerate(self.message_blocks):

            # message block

            ds_message_ij, dv_message_ij = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs)


            dv_message = scatter_add(src=dv_message_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

            ds_message = scatter_add(src=ds_message_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            # update block
            update_block = self.update_blocks[i]
            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)
            s_i = s_i + ds_update
            v_i = v_i + dv_update

            # edge block
            edge_update_block = self.edge_update[i]
            de_update = edge_update_block(ds_message_ij)
            e_ij = e_ij + de_update

        f_edge = self.edgereadout(e_ij) * xyz_adjoint
        f_atom = scatter_add(f_edge, nbrs[:,0], dim=0, dim_size=graph_size)

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

