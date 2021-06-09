import torch
from torch import nn

from nff.utils.tools import layer_types
from nff.nn.layers import (PainnRadialBasis, CosineEnvelope,
                           ExpNormalBasis, Dense)
from nff.utils.scatter import scatter_add
from nff.nn.modules.schnet import ScaleShift
from nff.nn.modules.torchmd_net import MessageBlock as MDMessage
from nff.nn.modules.torchmd_net import EmbeddingBlock as MDEmbedding

EPS = 1e-15


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


def to_module(activation):
    return layer_types[activation]()


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

        feat_dim = s_j.shape[-1]
        out_reshape = output.reshape(output.shape[0], 3, feat_dim)

        return out_reshape


class MessageBase(nn.Module):

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

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class MessageBlock(MessageBase):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout,
                 **kwargs):
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
                nbrs,
                **kwargs):

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

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class InvariantTransformerMessage(nn.Module):
    def __init__(self,
                 rbf,
                 num_heads,
                 feat_dim,
                 activation,
                 layer_norm):

        super().__init__()
        self.msg_layer = MDMessage(feat_dim=feat_dim,
                                   num_heads=num_heads,
                                   activation=activation,
                                   rbf=rbf)

        self.dense = Dense(in_features=(num_heads * feat_dim),
                           out_features=(3 * feat_dim),
                           bias=True,
                           activation=None)
        self.layer_norm = nn.LayerNorm(feat_dim) if (layer_norm) else None

    def forward(self,
                s_j,
                dist,
                nbrs):

        inp = self.layer_norm(s_j) if self.layer_norm else s_j
        output = self.dense(self.msg_layer(dist=dist,
                                           nbrs=nbrs,
                                           x_i=inp))
        out_reshape = output.reshape(output.shape[0], 3, -1)
        return out_reshape


class TransformerMessageBlock(MessageBase):
    def __init__(self,
                 rbf,
                 num_heads,
                 feat_dim,
                 activation,
                 layer_norm):
        super().__init__()

        self.inv_message = InvariantTransformerMessage(
            rbf=rbf,
            num_heads=num_heads,
            feat_dim=feat_dim,
            activation=activation,
            layer_norm=layer_norm)


class UpdateBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout):
        super().__init__()
        self.u_mat = Dense(in_features=feat_dim,
                           out_features=feat_dim,
                           bias=False)
        self.v_mat = Dense(in_features=feat_dim,
                           out_features=feat_dim,
                           bias=False)
        self.s_dense = nn.Sequential(Dense(in_features=2*feat_dim,
                                           out_features=feat_dim,
                                           bias=True,
                                           dropout_rate=dropout,
                                           activation=to_module(activation)),
                                     Dense(in_features=feat_dim,
                                           out_features=3*feat_dim,
                                           bias=True,
                                           dropout_rate=dropout))

    def forward(self,
                s_i,
                v_i):

        # v_i = (num_atoms, num_feats, 3)
        # v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        # = (num_atoms, 3, num_feats).reshape(-1, num_feats)
        # = (num_atoms * 3, num_feats)
        # -> So the same u gets applied to each atom
        # and for each of the three dimensions, but differently
        # for the different feature dimensions

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        # now reshape it to (num_atoms, 3, num_feats) and transpose
        # to get (num_atoms, num_feats, 3)

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = norm(v_v)
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = (self.s_dense(s_stack)
                 .reshape(s_i.shape[0], 3, -1))

        # delta v update
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class EmbeddingBlock(nn.Module):
    def __init__(self,
                 feat_dim):

        super().__init__()
        self.atom_embed = nn.Embedding(100, feat_dim, padding_idx=0)
        self.feat_dim = feat_dim

    def forward(self,
                z_number,
                **kwargs):

        num_atoms = z_number.shape[0]
        s_i = self.atom_embed(z_number)
        v_i = (torch.zeros(num_atoms, self.feat_dim, 3)
               .to(s_i.device))

        return s_i, v_i


class NbrEmbeddingBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 dropout,
                 rbf):

        super().__init__()
        self.embedding = MDEmbedding(feat_dim=feat_dim,
                                     dropout=dropout,
                                     rbf=rbf)
        self.feat_dim = feat_dim

    def forward(self,
                z_number,
                nbrs,
                r_ij):

        num_atoms = z_number.shape[0]

        dist, _ = preprocess_r(r_ij)
        s_i = self.embedding(z_number=z_number,
                             nbrs=nbrs,
                             dist=dist)

        v_i = (torch.zeros(num_atoms, self.feat_dim, 3)
               .to(s_i.device))

        return s_i, v_i


class ReadoutBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 output_keys,
                 activation,
                 dropout,
                 means=None,
                 stddevs=None):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {key: nn.Sequential(
                Dense(in_features=feat_dim,
                      out_features=feat_dim//2,
                      bias=True,
                      dropout_rate=dropout,
                      activation=to_module(activation)),
                Dense(in_features=feat_dim//2,
                      out_features=1,
                      bias=True,
                      dropout_rate=dropout))
             for key in output_keys}
        )

        self.scale_shift = ScaleShift(means=means,
                                      stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            output = self.scale_shift(output, key)
            results[key] = output

        return results
