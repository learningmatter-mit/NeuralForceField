import torch
from torch import nn

from nff.utils.scatter import scatter_add
from nff.utils.tools import layer_types
from nff.nn.layers import Dense


def get_cat_dim(atom_embed_dim, n_rbf):
    return 2 * atom_embed_dim + n_rbf


def get_dense(inp_dim, out_dim, activation, bias):
    activation = layer_types[activation]()
    return Dense(inp_dim, out_dim, activation=activation, bias=bias)


class EdgeEmbedding(nn.Module):
    def __init__(self, atom_embed_dim, n_rbf, activation):
        super().__init__()
        cat_dim = get_cat_dim(atom_embed_dim, n_rbf)
        self.dense = get_dense(cat_dim, cat_dim, activation=activation, bias=True)

    def forward(self, h, e, nbr_list):
        cat = torch.cat((h[nbr_list[:, 0]], h[nbr_list[:, 1]], e), dim=-1)
        m_ji = self.dense(cat)
        return m_ji


class NodeEmbedding(nn.Module):
    def __init__(self, atom_embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(100, atom_embed_dim, padding_idx=0)

    def forward(self, z):
        out = self.embedding(z)
        return out


class EmbeddingBlock(nn.Module):
    def __init__(self, n_rbf, atom_embed_dim, activation):
        super().__init__()
        self.edge_linear = nn.Linear(n_rbf, n_rbf, bias=False)
        self.node_embedding = NodeEmbedding(atom_embed_dim)
        self.cat_embedding = EdgeEmbedding(atom_embed_dim,
                                           n_rbf,
                                           activation)

    def forward(self, e_rbf, z, nbr_list):
        e = self.edge_linear(e_rbf)
        h = self.node_embedding(z)
        out = self.cat_embedding(h=h,
                                 e=e,
                                 nbr_list=nbr_list)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, atom_embed_dim, n_rbf, activation):
        super().__init__()
        cat_dim = get_cat_dim(atom_embed_dim, n_rbf)
        self.dense_layers = nn.ModuleList(
            [get_dense(cat_dim, cat_dim, activation=activation, bias=True)
                for _ in range(2)]
        )

    def forward(self, m_ji):

        out = m_ji.clone()
        for layer in self.dense_layers:
            out = self.dense_layers(out)

        return out + m_ji


class DirectedMessage(nn.Module):
    def __init__(self,
                 activation,
                 atom_embed_dim,
                 n_rbf,
                 n_spher,
                 l_spher,
                 n_bilinear):

        super().__init__()

        cat_dim = get_cat_dim(atom_embed_dim, n_rbf)
        a_dim = n_spher * l_spher

        self.sigma = layer_types[activation]()
        self.nbr_m_linear = nn.Linear(cat_dim, cat_dim, bias=True)
        # is this the right dimensionality?
        self.e_rbf_linear = nn.Linear(n_rbf, cat_dim, bias=False)

        self.a_sbf_linear = nn.Linear(a_dim, n_bilinear, bias=False)
        self.final_w = nn.Parameter(torch.empty(n_rbf, cat_dim))
        nn.init.xavier_uniform_(self.final_w)

    def nbr_m_block(self, m_ji, kj_idx):

        m_kj = m_ji[kj_idx]
        out = self.sigma(self.nbr_m_linear(m_kj))

        return out

    def e_block(self, e_rbf, kj_idx):

        repeated_e = e_rbf[kj_idx]
        out = self.e_rbf_linear(repeated_e)

        return out

    def nbr_m_and_e(self, m_ji, nbr_list, angle_list, e_rbf,
                    kj_idx):

        transf_nbr_m = self.nbr_m_block(m_ji, kj_idx)
        transf_e_rbf = self.e_block(e_rbf, kj_idx)

        out = transf_nbr_m * transf_e_rbf

        return out, kj_idx

    def forward(self, m_ji, nbr_list, angle_list, e_rbf, a_sbf,
                kj_idx):
        m_and_e = self.nbr_m_and_e(m_ji=m_ji,
                                   nbr_list=nbr_list,
                                   angle_list=angle_list,
                                   e_rbf=e_rbf)
        transf_a = self.a_sbf_linear(a_sbf)
        # check this
        out = (transf_a * torch.matmul(self.final_w,
                                       m_and_e)) * sum(-1)

        # sum over k

        final = scatter_add(out, kj_idx, m_ji.shape[0])

        return final


class InteractionBlock(nn.Module):
    def __init__(self,
                 atom_embed_dim,
                 n_rbf,
                 activation,
                 n_spher,
                 l_spher,
                 n_bilinear):
        super().__init__()

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(
                atom_embed_dim=atom_embed_dim,
                n_rbf=n_rbf,
                activation=activation) for _ in range(3)]
        )

        self.directed_block = DirectedMessage(
            activation=activation,
            atom_embed_dim=atom_embed_dim,
            n_rbf=n_rbf,
            n_spher=n_spher,
            l_spher=l_spher,
            n_bilinear=n_bilinear)

        cat_dim = get_cat_dim(atom_embed_dim, n_rbf)
        self.m_ji_dense = get_dense(
            cat_dim, cat_dim, activation=activation, bias=True)
        self.post_res_dense = get_dense(
            cat_dim, cat_dim, activation=activation, bias=True)

    def forward(self, m_ji, nbr_list, angle_list, e_rbf, a_sbf,
                kj_idx):
        directed_out = self.directed_block(m_ji=m_ji,
                                           nbr_list=nbr_list,
                                           angle_list=angle_list,
                                           e_rbf=e_rbf,
                                           a_sbf=a_sbf,
                                           kj_idx=kj_idx)
        transf_m_ji = self.m_ji_dense(m_ji)
        output = directed_out + transf_m_ji
        output = self.post_res_dense(
            self.residual_blocks[0](output)) + transf_m_ji
        for res_block in self.residual_blocks[1:]:
            output = res_block(output)

        return output


class OutputBlock(nn.Module):
    def __init__(self,
                 atom_embed_dim,
                 n_rbf,
                 activation):
        super().__init__()

        cat_dim = get_cat_dim(atom_embed_dim, n_rbf)
        self.edge_linear = nn.Linear(n_rbf, cat_dim, bias=False)
        self.dense_layers = nn.ModuleList(
            [
                get_dense(cat_dim, cat_dim, activation=activation, bias=True)
                for _ in range(3)
            ])
        self.final_linear = nn.Linear(cat_dim, cat_dim, bias=False)

    def forward(self, m_ji, e_rbf, nbr_list, num_atoms):
        prod = self.edge_linear(e_rbf) * m_ji
        node_feats = scatter_add(prod, nbr_list[:, 0], num_atoms)
        for dense in self.dense_layers:
            node_feats = dense(node_feats)
        node_feats = self.final_linear(node_feats)

        return node_feats
