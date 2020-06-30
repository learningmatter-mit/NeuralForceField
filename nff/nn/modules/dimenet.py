import torch
from torch import nn

from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types
from nff.nn.layers import Dense


def get_dense(inp_dim, out_dim, activation, bias):
    if activation is not None:
        activation = layer_types[activation]()
    return Dense(inp_dim, out_dim, activation=activation, bias=bias)


class EdgeEmbedding(nn.Module):
    def __init__(self, embed_dim, n_rbf, activation):
        super().__init__()
        self.dense = get_dense(
            3 * embed_dim,
            embed_dim,
            activation=activation,
            bias=True)

    def forward(self, h, e, nbr_list):
        m_ji = torch.cat((h[nbr_list[:, 0]], h[nbr_list[:, 1]], e), dim=-1)
        m_ji = self.dense(m_ji)
        return m_ji


class NodeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(100, embed_dim, padding_idx=0)

    def forward(self, z):
        out = self.embedding(z)
        return out


class EmbeddingBlock(nn.Module):
    def __init__(self, n_rbf, embed_dim, activation):
        super().__init__()
        self.edge_dense = get_dense(n_rbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)
        self.node_embedding = NodeEmbedding(embed_dim)
        self.edge_embedding = EdgeEmbedding(embed_dim,
                                            n_rbf,
                                            activation)

    def forward(self, e_rbf, z, nbr_list):
        e = self.edge_dense(e_rbf)
        h = self.node_embedding(z)
        m_ji = self.edge_embedding(h=h,
                                   e=e,
                                   nbr_list=nbr_list)
        return m_ji


class ResidualBlock(nn.Module):
    def __init__(self, embed_dim, n_rbf, activation):
        super().__init__()
        self.dense_layers = nn.ModuleList(
            [get_dense(embed_dim,
                       embed_dim,
                       activation=activation,
                       bias=True)
                for _ in range(2)]
        )

    def forward(self, m_ji):

        residual = m_ji.clone()
        for layer in self.dense_layers:
            residual = layer(residual)

        return residual + m_ji


class DirectedMessage(nn.Module):
    def __init__(self,
                 activation,
                 embed_dim,
                 n_rbf,
                 n_spher,
                 l_spher,
                 n_bilinear):

        super().__init__()

        self.m_kj_dense = get_dense(embed_dim,
                                    embed_dim,
                                    activation=activation,
                                    bias=True)
        self.e_dense = get_dense(n_rbf,
                                 embed_dim,
                                 activation=None,
                                 bias=False)
        self.a_dense = get_dense(n_spher * l_spher,
                                 n_bilinear,
                                 activation=None,
                                 bias=False)
        self.w = nn.Parameter(torch.empty(
            embed_dim, n_bilinear, embed_dim))

        nn.init.xavier_uniform_(self.w)

    def forward(self,
                m_ji,
                e_rbf,
                a_sbf,
                kj_idx,
                ji_idx):

        e_kj = self.e_dense(e_rbf[kj_idx])
        m_kj = self.m_kj_dense(m_ji[kj_idx])
        a = self.a_dense(a_sbf)

        # Is this right? Here's where we really have to
        # be careful about kj, jk, ij, etc.

        aggr = torch.einsum("wj,wl,ijl->wi", a, m_kj * e_kj, self.w)

        # For example, aggr = {aggr_kj,ji} = [aggr_{0,1,2}, aggr_{0,1,3}].
        # = [aggr_{21,10}, aggr_{31,10}].
        # What we want is to sum out all k's that correspond to the same
        # {j, i}, i.e. those for which indices 0 and 1 are the same.
        # We want the sum to be in an array at index l, where l is the
        # index at which nbr_list[l][0] = j and nbr_list[l][1] = i.
        # Say nbr_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]].
        # Then the 1,0 index is equal to 2, and so we want to sum these
        # at index 2. This is precisely what we get from "ji_idx".

        out = scatter_add(aggr.transpose(0, 1),
                          ji_idx,
                          dim_size=m_ji.shape[0]
                          ).transpose(0, 1)

        return out


class InteractionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_rbf,
                 activation,
                 n_spher,
                 l_spher,
                 n_bilinear):
        super().__init__()

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(
                embed_dim=embed_dim,
                n_rbf=n_rbf,
                activation=activation) for _ in range(3)]
        )

        self.directed_block = DirectedMessage(
            activation=activation,
            embed_dim=embed_dim,
            n_rbf=n_rbf,
            n_spher=n_spher,
            l_spher=l_spher,
            n_bilinear=n_bilinear)

        self.m_ji_dense = get_dense(embed_dim,
                                    embed_dim,
                                    activation=activation,
                                    bias=True)

        self.post_res_dense = get_dense(embed_dim,
                                        embed_dim,
                                        activation=activation,
                                        bias=True)

    def forward(self, m_ji, nbr_list, angle_list, e_rbf, a_sbf,
                kj_idx, ji_idx):
        directed_out = self.directed_block(m_ji=m_ji,
                                           e_rbf=e_rbf,
                                           a_sbf=a_sbf,
                                           kj_idx=kj_idx,
                                           ji_idx=ji_idx)
        dense_m_ji = self.m_ji_dense(m_ji)
        output = directed_out + dense_m_ji
        output = self.post_res_dense(
            self.residual_blocks[0](output)) + m_ji
        for res_block in self.residual_blocks[1:]:
            output = res_block(output)

        return output


class OutputBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_rbf,
                 activation):
        super().__init__()

        self.edge_dense = get_dense(n_rbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)

        self.dense_layers = nn.ModuleList(
            [
                get_dense(embed_dim,
                          embed_dim,
                          activation=activation,
                          bias=True)
                for _ in range(3)
            ])
        self.dense_layers.append(get_dense(embed_dim,
                                           embed_dim,
                                           activation=None,
                                           bias=False))

    def forward(self, m_ji, e_rbf, nbr_list, num_atoms):
        prod = self.edge_dense(e_rbf) * m_ji

        # the messages are m = {m_ji} =, for example, 
        # [m_{0,1}, m_{0,2}], with nbr_list = [[0, 1], [0, 2]]. 
        # To sum over the j index we would have the first of
        # these messages add to index 1 and the second to index 2.
        # That means we use
        # nbr_list[:, 1] in the scatter addition.

        node_feats = scatter_add(prod.transpose(0, 1),
                                 nbr_list[:, 1],
                                 dim_size=num_atoms).transpose(0, 1)

        for dense in self.dense_layers:
            node_feats = dense(node_feats)

        return node_feats
