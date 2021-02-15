import torch
from torch import nn

from nff.utils.tools import layer_types
from nff.nn.layers import PainnRadialBasis as RadialBasis
from nff.utils.scatter import scatter_add, compute_grad

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


class InvariantDense(nn.Module):
    def __init__(self, dim, activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=dim,
                                              out_features=dim,
                                              bias=True),
                                    layer_types[activation](),
                                    nn.Linear(in_features=dim,
                                              out_features=3 * dim,
                                              bias=True))

    def forward(self, s_j):
        output = self.layers(s_j)
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

        s_j_nbrs = s_j[nbrs[:, 1]]
        phi = self.inv_dense(s_j_nbrs)
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

        split_0 = inv_out[:, :, 0].reshape(-1, s_j.shape[1], 1)
        split_1 = inv_out[:, :, 1]
        split_2 = inv_out[:, :, 2].reshape(-1, s_j.shape[1], 1)

        unit_add = split_2 * unit.reshape(-1, 1, 3)
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


class UpdateBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation):
        super().__init__()
        self.u_mat = nn.Linear(in_features=feat_dim,
                               out_features=feat_dim,
                               bias=False)
        self.v_mat = nn.Linear(in_features=feat_dim,
                               out_features=feat_dim,
                               bias=False)
        self.s_dense = nn.Sequential(nn.Linear(in_features=2*feat_dim,
                                               out_features=feat_dim,
                                               bias=True),
                                     layer_types[activation](),
                                     nn.Linear(in_features=feat_dim,
                                               out_features=3*feat_dim,
                                               bias=True))

    def forward(self,
                s_i,
                v_i):

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, v_i.shape[1])
               .transpose(1, 2))

        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, v_i.shape[1])
               .transpose(1, 2))

        v_v_norm = norm(v_v)
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)
        split = (self.s_dense(s_stack)
                 .reshape(s_i.shape[0], -1, 3))

        # delta v update
        a_vv = split[:, :, 0].reshape(*split.shape[:2], 1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, :, 1]
        a_ss = split[:, :, 2]
        # check this
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
                z_number):

        num_atoms = z_number.shape[0]
        s_i = self.atom_embed(z_number)
        v_i = (torch.zeros(num_atoms, self.feat_dim, 3)
               .to(s_i.device))

        return s_i, v_i


class ReadoutBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 output_keys,
                 grad_keys,
                 activation):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {key: nn.Sequential(
                nn.Linear(in_features=feat_dim,
                          out_features=feat_dim//2,
                          bias=True),
                layer_types[activation](),
                nn.Linear(in_features=feat_dim//2,
                          out_features=1,
                          bias=True)
            )
                for key in output_keys}
        )

        self.grad_keys = grad_keys

    def forward(self,
                s_i,
                xyz,
                num_atoms):

        output = {key: self.readoutdict[key](s_i)
                  for key in self.readoutdict.keys()}
        results = {}

        for key, val in output.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, num_atoms)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        for key in self.grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
