import torch
from torch import nn
from torch.nn import Sequential 
import numpy as np
import copy

from nff.utils.tools import make_directed
from nff.nn.modules.painn import (DistanceEmbed, MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock, 
                                  to_module, norm)
from nff.nn.modules.schnet import (AttentionPool, SumPool)
# from nff.nn.modules.diabat import DiabaticReadout
# from nff.nn.layers import Diagonalize
from nff.nn.layers import GaussianSmearing, Dense, PainnRadialBasis, CosineEnvelope
from nff.nn.activations import shifted_softplus
from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types


# POOL_DIC = {"sum": SumPool,
#             "attention": AttentionPool}

class DistEmb(nn.Module):
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
                      out_features=feat_dim,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


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
        self.dist_embed = DistEmb(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=feat_dim,
                                  learnable_k=learnable_k,
                                  dropout=conv_dropout)
        
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
                                 out_features=2*feat_dim,
                                 bias=True,
                                 dropout_rate=conv_dropout,
                                 activation=to_module(activation)), 
                           Dense(in_features=2*feat_dim, 
                                 out_features=feat_dim,
                                 bias=True,
                                 dropout_rate=conv_dropout))
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
                  out_features=feat_dim*2, 
                  bias=True, 
                  dropout_rate=readout_dropout,
                  activation=to_module(activation)),
            Dense(in_features=feat_dim*2, 
                  out_features=1, 
                  bias=True,
                  dropout_rate=readout_dropout)
        )
        # self.edgereadout = EdgeReadoutBlock(feat_dim=feat_dim,
        #                                     activation=activation,
        #                                     dropout=readout_dropout)
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
        dis_vec = r_ij
        dis = norm(r_ij)
        xyz_adjoint = dis_vec / dis.unsqueeze(-1)
        e_ij = self.dist_embed(dis)

        results = {}

        nconv = len(self.message_blocks)
        for i, message_block in enumerate(self.message_blocks):

            # message block
            ds_message_ij, dv_message_ij = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs,
                                                   e_ij=None)
            
            graph_size = s_i.shape[0]

            ds_message = scatter_add(src=ds_message_ij,
                                    index=nbrs[:, 0],
                                    dim=0,
                                    dim_size=graph_size)

            dv_message = scatter_add(src=dv_message_ij,
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
            de_update = edge_update_block(s_i[nbrs[:, 0]] + s_i[nbrs[:, 1]])
            e_ij = e_ij + de_update

        f_edge = self.edgereadout(e_ij) * xyz_adjoint
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



class ForcePai_polar(nn.Module):
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
        self.dist_embed = DistEmb(n_rbf=n_rbf,
                                  cutoff=cutoff,
                                  feat_dim=feat_dim,
                                  learnable_k=learnable_k,
                                  dropout=conv_dropout)
        
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

        self.edge_update = nn.ModuleList(
            [nn.Sequential(Dense(in_features=feat_dim, 
                                 out_features=2*feat_dim,
                                 bias=True,
                                 dropout_rate=conv_dropout,
                                 activation=to_module(activation)), 
                           Dense(in_features=2*feat_dim, 
                                 out_features=feat_dim,
                                 bias=True,
                                 dropout_rate=conv_dropout))
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
                  out_features=feat_dim*2, 
                  bias=True, 
                  dropout_rate=readout_dropout,
                  activation=to_module(activation)),
            Dense(in_features=feat_dim*2, 
                  out_features=1, 
                  bias=True,
                  dropout_rate=readout_dropout)
        )

    def atomwise(self,
                 batch,
                 xyz=None):
        
        '''
            https://lammps.sandia.gov/doc/pair_dipole.html
        '''

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
        dis_vec = r_ij
        dis = norm(r_ij)
        xyz_adjoint = dis_vec / dis.unsqueeze(-1)
        e_ij = self.dist_embed(dis)

        results = {}

        for i, message_block in enumerate(self.message_blocks):

            # message block
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs,
                                                   e_ij=None)
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
            de_update = edge_update_block( (v_i[nbrs[:, 0]] * v_i[nbrs[:, 1]]).sum(-1) )
            
            fqq = (s_i[nbrs[:, 0]] * s_i[nbrs[:, 1]]).mean(-1)[..., None] * xyz_adjoint 
            
            
            fqp1 = -(s_i[nbrs[:, 0]].unsqueeze(-1) * v_i[nbrs[:, 1]]).mean(1) / dis.unsqueeze(-1)
            
            pr_dot = v_i[nbrs[:,1]] * xyz_adjoint.unsqueeze(1)

            fqp2 = 3 * (s_i[nbrs[:, 0]].unsqueeze(-1) * pr_dot).mean(1).sum(-1).unsqueeze(-1) * xyz_adjoint / dis.unsqueeze(-1) 
            
            if  i== 0:
                f_edge = fqq + fqp1 + fqp2 #+ fpp
            else: 
                f_edge += fqq + fqp1 + fqp2 #+ fpp   
            
        
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
