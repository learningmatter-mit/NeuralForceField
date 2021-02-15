from torch import nn

from nff.utils.tools import layer_types
from nff.nn.layers import PainnRadialBasis as RadialBasis
from nff.utils.scatter import scatter_add


def preprocess_r(r_ij):
    """
    r_ij (n_nbrs x 3): tensor of interatomic vectors (r_j - r_i)
    """

    dist = (r_ij ** 2).sum(-1) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit


class InvariantDense(nn.Module):
    def __init__(self, dim, activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=dim,
                                              out_features=dim,
                                              bias=True),
                                    layer_types[activation],
                                    nn.Linear(in_features=dim,
                                              out_features=3 * dim,
                                              bias=True))

    def forward(self, s):
        output = self.layers(s)
        return output


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 feat_dim):

        super().__init__()
        rbf = RadialBasis(n_rbf=n_rbf, cutoff=cutoff)
        dense = nn.Linear(in_features=n_rbf,
                          out_features=3 * feat_dim,
                          bias=True)
        # what is this??
        f_cut = nn.Sequential()
        ##

        self.block = nn.Sequential(rbf, dense, f_cut)

    def forward(self, dist):
        output = self.block(dist)
        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff):
        super().__init__()

        self.inv_dense = InvariantDense(dim=feat_dim,
                                        activation=activation)
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim)

    def forward(self,
                s_j,
                dist,
                nbrs):
        phi = self.inv_dense(s_j[nbrs[:, 1]])
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x feat_dim x 3
        out_reshape = output.reshape(output.shape[0], -1, 3)

        return out_reshape


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff):
        super().__init__()
        self.inv_message = InvariantMessage(feat_dim=feat_dim,
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        equiv = (torch.stack([inv_out[:, 2]] * 3, dim=-1)
                 * unit.reshape(-1, 1, 3))
        delta_v_ij = equiv + (torch.stack([inv_out[:, 0]] * 3, dim=-1)
                              * v_j)
        delta_s_ij = inv_out[:, 1]

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
