import torch
from torch import nn

from nff.utils.tools import layer_types
from nff.nn.layers import PainnRadialBasis, CosineEnvelope
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

    # import pdb
    # pdb.set_trace()

    # for ethanol
    # print(r_ij[0])
    # print(r_ij[8])

    return dist, unit


class InvariantDense(nn.Module):
    def __init__(self,
                 dim,
                 activation='swish'):
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
                 feat_dim,
                 learnable_k):

        super().__init__()
        self.rbf = PainnRadialBasis(n_rbf=n_rbf,
                                    cutoff=cutoff,
                                    learnable_k=learnable_k)
        self.dense = nn.Linear(in_features=n_rbf,
                               out_features=3 * feat_dim,
                               bias=True)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.dense(self.rbf(dist))
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k):
        super().__init__()

        self.inv_dense = InvariantDense(dim=feat_dim,
                                        activation=activation)
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=learnable_k)

    def forward(self,
                s_j,
                dist,
                nbrs):

        # s_j_nbrs = s_j[nbrs[:, 1]]

        # import pdb
        # pdb.set_trace()

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x feat_dim
        out_reshape = output.reshape(output.shape[0], 3, -1)

        # import pdb
        # pdb.set_trace()
        # print(output[5, 3])
        # print(out_reshape[5, 3, 0])

        # print(output[5, 133])
        # print(out_reshape[5, 5, 1])

        return out_reshape


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k):
        super().__init__()
        self.inv_message = InvariantMessage(feat_dim=feat_dim,
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            learnable_k=learnable_k)

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

        # import pdb
        # pdb.set_trace()
        
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

        # # testing

        # test = torch.stack([i * (r_ij[3] / r_ij[3].norm()) for i in inv_out[3, :, 2]])
        # real = unit_add[3]
        # print(abs(real - test).max())

        # # because nbrs[3] = [0, 4]
        # test_2 = v_j[4] * inv_out[3, :, 0].reshape(-1, 1)
        # final_test = test + test_2
        # final_real = real + (split_0 * v_j[nbrs[:, 1]])[3]

        # print(abs(final_real - final_test).max())

        # import pdb
        # pdb.set_trace()

        # nbrs_of_0_idx = (nbrs[:, 0] == 0).nonzero()
        # test_0 = delta_v_ij[nbrs_of_0_idx].sum(0)

        # print(abs(delta_v_i[0] - test_0).max())

        # nbrs_of_1_idx = (nbrs[:, 0] == 1).nonzero()
        # test_1 = delta_s_ij[nbrs_of_1_idx].sum(0)
        # print(abs(delta_s_i[1] - test_1).max())

        # import pdb
        # pdb.set_trace()

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

        # v_i = (num_atoms, num_feats, 3)
        # v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        # = (num_atoms, 3, num_feats).reshape(-1, num_feats)
        # = (num_atoms * 3, num_feats)
        # -> So the same u gets applied to to each atom
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

        # Testing
        # Test equivariance:

        # # take u_v of atom 1 for feature dimension 4
        # # and rotate it
        # test_A = u_v[1, 4]
        # rot = torch.Tensor([[0.9752, -0.1449,  0.1675],
        #                     [0.1977,  0.9104, -0.3634],
        #                     [-0.0998,  0.3875,  0.9165]]
        #                    ).to(test_A.device)
        # out_A = torch.matmul(rot, test_A)

        # # rotate v of atom 1 for each of the feature dimensions
        # # and then stack
        # test_B = torch.stack([torch.matmul(rot, i)
        #                       for i in v_i[1]])

        # # multiply each vector k by the weight w_4k
        # # and sum

        # u_mat = next(iter(self.u_mat.parameters()))
        # u_vec = u_mat[4, :]
        # out_B = (u_vec.reshape(-1, 1) * test_B).sum(0)

        # import pdb
        # pdb.set_trace()

        # print((abs(out_A - out_B).max()))

        # import pdb
        # pdb.set_trace()

        # split_0_3 = split[:, :, 0][3]
        # u_v_3 = u_v[3]
        # test = split_0_3[5] * u_v_3[5]
        # real = delta_v_i[3, 5]
        # print(abs(test - real).max().item())

        # inner_test = inner[10, 13] * split[10, 13, 1] + split[10, 13, 2]
        # inner_real = delta_s_i[10, 13]

        # print(abs(inner_real - inner_test).item())

        # test_inner_at_5_feat_3 = (u_v[5, 3, :] * v_v[5, 3, :]).sum()
        # real_inner_at_5_feat_3 = inner[5, 3]
        # print(abs(test_inner_at_5_feat_3 -
        #           real_inner_at_5_feat_3).max().detach().item())

        # # test_u_v_1 = self.u_mat(v_i[:, :, 1])
        # # real_u_v_1 = u_v[:, :, 1]
        # # print(abs(test_u_v_1 - real_u_v_1).max().detach().item())

        # # test_v_v_1 = self.v_mat(v_i[:, :, 1])
        # # real_v_v_1 = v_v[:, :, 1]
        # # print(abs(test_v_v_1 - real_v_v_1).max().detach().item())

        # # u_params = next(iter(self.u_mat.parameters()))
        # # v_params = next(iter(self.v_mat.parameters()))

        # # u_v_test = torch.stack([torch.matmul(u_params, v_i[3, :, 0]),
        # #                         torch.matmul(u_params, v_i[3, :, 1]),
        # #                         torch.matmul(u_params, v_i[3, :, 2])]
        # #                        ).transpose(0, 1)

        # # v_v_test = torch.stack([torch.matmul(v_params, v_i[5, :, 0]),
        # #                         torch.matmul(v_params, v_i[5, :, 1]),
        # #                         torch.matmul(v_params, v_i[5, :, 2])]
        # #                        ).transpose(0, 1)

        # import pdb
        # pdb.set_trace()

        # # print(abs(u_v_test - u_v[3, :]).max().detach().item())
        # # print(abs(v_v_test - v_v[5, :]).max().detach().item())

        # # print(abs(s_stack[3, :4] - s_i[3, :4]).max().detach())
        # # print(abs(s_stack[3, -4:] - v_v_norm[3, -4:]).max().detach())

        # v_v_norm_test = v_v[1, 4].norm()
        # v_v_norm_real = v_v_norm[1, 4]
        # print(abs(v_v_norm_real - v_v_norm_test).max().item())

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
                          bias=True))
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

            # import pdb
            # pdb.set_trace()
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        for key in self.grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
