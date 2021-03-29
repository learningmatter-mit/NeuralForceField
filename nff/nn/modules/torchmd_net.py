import torch
from torch import nn
from nff.nn.layers import (CosineEnvelope, Dense)
from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types


class DistanceEmbeding(nn.Module):

    def __init__(self,
                 feat_dim,
                 dropout,
                 rbf,
                 bias=False):

        super().__init__()

        n_rbf = rbf.mu.shape[0]
        cutoff = rbf.cutoff
        dense = Dense(in_features=n_rbf,
                      out_features=feat_dim,
                      bias=bias,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class EmbeddingBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 dropout,
                 rbf):

        super().__init__()
        self.atom_embed = nn.Embedding(100, feat_dim, padding_idx=0)
        self.feat_dim = feat_dim
        self.distance_embed = DistanceEmbeding(feat_dim=feat_dim,
                                               dropout=dropout,
                                               rbf=rbf,
                                               bias=False)

        self.concat_dense = Dense(in_features=2*feat_dim,
                                  out_features=feat_dim,
                                  dropout_rate=dropout,
                                  activation=None)

    def forward(self,
                z_number,
                nbrs,
                dist):

        num_atoms = z_number.shape[0]
        node_embeddings = self.atom_embed(z_number)

        nbr_embeddings = self.atom_embed(z_number[nbrs[:, 1]])
        edge_feats = self.distance_embed(dist) * nbr_embeddings
        aggr_embeddings = scatter_add(src=edge_feats,
                                      index=nbrs[:, 0],
                                      dim=0,
                                      dim_size=num_atoms)

        final_embeddings = self.concat_dense(torch.cat([node_embeddings,
                                                        aggr_embeddings],
                                                       dim=-1))

        return final_embeddings


class AttentionHeads(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 num_heads,
                 rbf):

        super().__init__()

        self.rbf = rbf
        n_rbf = rbf.mu.shape[0]
        cutoff = rbf.cutoff
        self.f_cut = CosineEnvelope(cutoff=cutoff)

        self.dk_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_rbf * num_heads,
                out_channels=feat_dim * num_heads,
                kernel_size=1,
                groups=num_heads),
            layer_types[activation]()
        )

        self.query_layer = nn.Conv1d(in_channels=feat_dim * num_heads,
                                     out_channels=feat_dim * num_heads,
                                     kernel_size=1,
                                     groups=num_heads)

        self.key_layer = nn.Conv1d(in_channels=feat_dim * num_heads,
                                   out_channels=feat_dim * num_heads,
                                   kernel_size=1,
                                   groups=num_heads)

        self.activation = layer_types[activation]()
        self.num_heads = num_heads

    def forward(self,
                dist,
                nbrs,
                x_i):

        x_i_nbrs = x_i[nbrs[:, 0]]
        x_j = x_i[nbrs[:, 1]]
        edge_feats = (self.rbf(dist)
                      * self.f_cut(dist).reshape(-1, 1))

        x_i_inp = x_i_nbrs.repeat(1, self.num_heads).unsqueeze(-1)
        x_j_inp = x_j.repeat(1, self.num_heads).unsqueeze(-1)
        edge_inp = edge_feats.repeat(1, self.num_heads).unsqueeze(-1)

        query = self.query_layer(x_i_inp)
        key = self.key_layer(x_j_inp)
        d_k = self.dk_layer(edge_inp)

        product = query * key * d_k

        prod_split = product.reshape(product.shape[0], self.num_heads, -1)
        weights = self.activation(prod_split.sum(dim=-1))

        return weights


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 num_heads,
                 activation,
                 rbf):

        super().__init__()

        n_rbf = rbf.mu.shape[0]
        self.attention_heads = AttentionHeads(feat_dim=feat_dim,
                                              activation=activation,
                                              num_heads=num_heads,
                                              rbf=rbf)

        self.v_layer = nn.Conv1d(in_channels=feat_dim * num_heads,
                                 out_channels=feat_dim * num_heads,
                                 kernel_size=1,
                                 groups=num_heads)

        self.dv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_rbf * num_heads,
                out_channels=feat_dim * num_heads,
                kernel_size=1,
                groups=num_heads),
            layer_types[activation]()
        )

        self.rbf = self.attention_heads.rbf
        self.f_cut = self.attention_heads.f_cut
        self.num_heads = num_heads

    def forward(self,
                dist,
                nbrs,
                x_i):

        x_j = x_i[nbrs[:, 1]]
        edge_feats = (self.rbf(dist)
                      * self.f_cut(dist).reshape(-1, 1))

        x_j_inp = x_j.repeat(1, self.num_heads).unsqueeze(-1)
        edge_inp = edge_feats.repeat(1, self.num_heads).unsqueeze(-1)

        v_feats = self.v_layer(x_j_inp)
        d_v = self.dv_layer(edge_inp)

        prod_v = v_feats * d_v

        weights = self.attention_heads(dist=dist,
                                       nbrs=nbrs,
                                       x_i=x_i)

        # dimension num_edges x num_heads x num_features
        scaled_v = (prod_v.reshape(prod_v.shape[0],
                                   self.num_heads, -1) *
                    weights.unsqueeze(-1))

        # reshape it into a concatenation, i.e. dimension
        # num_edges x (num_heads * num_features)

        scaled_v = scaled_v.reshape(scaled_v.shape[0], -1)

        return scaled_v


class UpdateBlock(nn.Module):
    def __init__(self,
                 num_heads,
                 feat_dim,
                 dropout):
        super().__init__()

        self.concat_dense = Dense(in_features=(num_heads * feat_dim),
                                  out_features=feat_dim,
                                  bias=True,
                                  dropout_rate=dropout,
                                  activation=None)

    def forward(self,
                nbrs,
                x_i,
                scaled_v):

        # dimension num_nodes x num_heads x num_features
        x_i_prime = scatter_add(src=scaled_v,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=x_i.shape[0])
        x_i = x_i + self.concat_dense(x_i_prime)

        return x_i
