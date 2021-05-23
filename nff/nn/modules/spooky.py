import torch
import numpy as np
from torch import nn
from nff.nn.layers import PreActivation, Dense
# from nff.nn.layers import (PainnRadialBasis, CosineEnvelope,
#                            ExpNormalBasis, Dense)
from nff.utils.tools import layer_types
from nff.utils.scatter import scatter_add
from nff.utils.constants import ELEC_CONFIG, KE_KCAL
from nff.utils import make_g_funcs

from performer_pytorch import FastAttention

EPS = 1e-15
DEFAULT_ACTIVATION = 'learnable_swish'
DEFAULT_MAX_Z = 86
DEFAULT_DROPOUT = 0


def norm(vec):
    result = ((vec ** 2 + EPS).sum(-1)) ** 0.5
    return result


def get_elec_config(max_z):
    max_z_config = torch.Tensor([max_z] + ELEC_CONFIG[max_z])
    # nan ensures we get nan results for any elements not
    # in ELEC_CONFIG
    elec_config = torch.ones(max_z + 1, 20) * float('nan')
    for z, val in ELEC_CONFIG.items():
        elec_config[z] = torch.Tensor([z] + val) / max_z_config
    return elec_config


class Residual(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 bias=True):

        super().__init__()
        self.block = nn.Sequential(PreActivation(in_features=feat_dim,
                                                 out_features=feat_dim,
                                                 activation=activation,
                                                 dropout_rate=dropout,
                                                 bias=bias),
                                   PreActivation(in_features=feat_dim,
                                                 out_features=feat_dim,
                                                 activation=activation,
                                                 dropout_rate=dropout,
                                                 bias=bias))

    def forward(self, x):
        output = x + self.block(x)
        return output


class ResidualMLP(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT):

        residual = Residual(feat_dim=feat_dim,
                            activation=activation,
                            dropout=dropout)
        self.block = nn.Sequential(residual,
                                   layer_types[activation](),
                                   nn.Linear(in_features=feat_dim,
                                             out_features=feat_dim))

    def forward(self, x):
        output = self.block(x)
        return output


class NuclearEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 max_z=DEFAULT_MAX_Z):

        super().__init__()
        self.elec_config = get_elec_config(max_z)
        self.m_mat = Dense(in_features=20,
                           out_features=feat_dim,
                           bias=False,
                           activation=None)
        self.z_embed = nn.Embedding(max_z, feat_dim, padding_idx=0)

    def forward(self, z):
        d_z = self.elec_config[z.long()]
        tilde_e_z = self.z_embed(z.long())
        e_z = self.m_mat(d_z) + tilde_e_z

        return e_z


class ElectronicEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.linear = Dense(in_features=feat_dim,
                            out_features=feat_dim,
                            bias=True,
                            activation=None)
        self.feat_dim = feat_dim
        self.res_mlp = ResidualMLP(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout,
                                   bias=False)
        names = ['k_plus', 'k_minus', 'v_plus', 'v_minus']
        for name in names:
            val = nn.Parameter(torch.ones(feat_dim)
                               .reshape(-1, 1))
            nn.init.xavier_uniform_(val)
            setattr(self, name, val)

    def forward(self,
                psi,
                e_z,
                num_atoms):

        q = self.linear(e_z)
        split_qs = torch.split(q, num_atoms)
        e_psi = torch.zeros_like(e_z)

        counter = 0

        for j, mol_q in enumerate(split_qs):
            mol_psi = psi[j]
            k = self.k_plus if (mol_psi >= 0) else self.k_minus
            # mol_q has dimension atoms_in_mol x F
            # k has dimension F x 1
            arg = (torch.einsum('ij, jk -> i', mol_q, k)
                   / self.feat_dim ** 0.5)
            num = torch.log(1 + torch.exp(arg))
            denom = num.sum()

            # dimension atoms_in_mol
            a_i = mol_psi * num / denom
            # dimension F x 1
            v = self.v_plus if (mol_psi >= 0) else self.v_minus

            # dimension atoms_in_mol x F
            av = v * a_i.reshape(-1, 1)
            this_e_psi = self.res_mlp(av)
            e_psi[counter: counter + num_atoms[j]] = this_e_psi
            counter += num_atoms[j]

        return e_psi


class CombinedEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z):

        super().__init__()
        self.nuc_embedding = NuclearEmbedding(
            feat_dim=feat_dim,
            max_z=max_z)
        self.charge_embedding = ElectronicEmbedding(
            feat_dim=feat_dim,
            activation=activation)
        self.spin_embedding = ElectronicEmbedding(feat_dim=feat_dim,
                                                  activation=activation)

    def forward(self,
                charge,
                spin,
                z,
                num_atoms):

        e_z = self.nuc_embedding(z)
        e_q = self.charge_embedding(psi=charge,
                                    e_z=e_z,
                                    num_atoms=num_atoms)
        e_s = self.charge_embedding(psi=spin,
                                    e_z=e_z,
                                    num_atoms=num_atoms)

        x_0 = e_z + e_q + e_s

        return x_0


class LocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 bern_k,
                 gamma,
                 r_cut,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z,
                 dropout=DEFAULT_DROPOUT):

        super().__init__()

        for letter in ["c", "s", "p", "d", "l"]:
            key = f"resmlp_{letter}"
            val = ResidualMLP(feat_dim=feat_dim,
                              activation=activation,
                              dropout=dropout)
            setattr(self, key, val)

        for key in ["G_s", "G_p", "G_d"]:
            val = nn.Parameter(torch.ones(feat_dim,
                                          bern_k))
            nn.init.xavier_uniform_(val)
            setattr(self, key, val)

        for key in ["P_1", "P_2", "D_1", "D_2"]:
            val = nn.Parameter(torch.ones(feat_dim,
                                          feat_dim))
            nn.init.xavier_uniform_(val)
            setattr(self, key, val)

        g_dic = make_g_funcs(bern_k=bern_k,
                             gamma=gamma,
                             r_cut=r_cut,
                             l_max=2)

        for key, val in g_dic.items():
            setattr(self, key, val)

    def g_matmul(self,
                 r_ij,
                 orbital):

        g_func = getattr(self, f"g_{orbital}")
        g = g_func(r_ij)
        G = getattr(self, f"G_{orbital}")
        # g: N_nbrs x K x (1, 3, or 5)
        # G: F x K
        # output: N_nbrs x F x (1, 3, or 5)

        out = torch.einsum('ik, jkl -> jil', G, g)
        return out

    def make_quant(self,
                   r_ij,
                   x_j,
                   nbrs,
                   graph_size,
                   orbital):

        res_block = getattr(self, f"resmlp_{orbital}")
        n_nbrs = nbrs.shape[0]

        matmul = self.g_matmul(r_ij, orbital)
        res_mlp = res_block(x_j)

        per_nbr = (res_mlp.reshape(n_nbrs, -1, 1)
                   * matmul)

        out = scatter_add(src=per_nbr,
                          index=nbrs[:, 0],
                          dim=0,
                          dim_size=graph_size)

        return out

    def take_inner(self,
                   quant,
                   orbital):

        name = orbital.upper()

        # dimensions F x F
        mat_1 = getattr(self, f"{name}_1")
        mat_2 = getattr(self, f"{name}_2")

        # quant has dimension n_atoms x F x (3 or 5)
        # term has dimension n_atoms x F x (3 or 5)

        term_1 = torch.einsum("ij, kjm->kim",
                              mat_1, quant)
        term_2 = torch.einsum("ij, kjm->kim",
                              mat_2, quant)

        # inner product multiplies elementwise and
        # sums over the last dimension

        inner = (term_1 * term_2).sum(-1)

        return inner

    def forward(self,
                xyz,
                x_tilde,
                nbrs):

        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]
        # dimension N_nbrs x F
        x_j = x_tilde[nbrs[:, 1]]
        graph_size = xyz.shape[0]

        c_term = self.resmlp_c(x_tilde)

        orbitals = ['s', 'p', 'd']
        quants = []
        for orbital in orbitals:
            quant = self.make_quant(r_ij=r_ij,
                                    x_j=x_j,
                                    nbrs=nbrs,
                                    graph_size=graph_size,
                                    orbital=orbital)
            quants.append(quant)

        invariants = []

        for quant, orbital in zip(quants[1:],
                                  orbitals[1:]):
            invariant = self.take_inner(quant,
                                        orbital)
            invariants.append(invariant)

        s_i = quants[0]
        s_term = s_i.reshape(graph_size, -1)
        inp = c_term + s_term + sum(invariants)
        out = self.resmlp_l(inp)

        return out


class NonLocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()

        # no redraw should happen here - only if
        # you call self attention of cross attention
        # as wrappers
        self.attn = FastAttention(dim_heads=feat_dim,
                                  nb_features=feat_dim,
                                  causal=False)
        self.feat_dim = feat_dim

        for letter in ['q', 'k', 'v']:
            key = f'resmlp_{letter}'
            val = ResidualMLP(feat_dim=feat_dim,
                              activation=activation,
                              dropout=dropout)
            setattr(self, key, val)

    def forward(self, x_tilde):
        # x_tilde has dimension N x F
        # N = number of nodes, F = feature dimension

        num_nodes = x_tilde.shape[0]
        Q = self.resmlp_q(x_tilde).reshape(1,
                                           1,
                                           num_nodes,
                                           self.feat_dim)
        K = self.resmlp_k(x_tilde).reshape(1,
                                           1,
                                           num_nodes,
                                           self.feat_dim)
        V = self.resmlp_v(x_tilde).reshape(1,
                                           1,
                                           num_nodes,
                                           self.feat_dim)

        out = self.attn(Q, K, V).reshape(num_nodes,
                                         self.feat_dim)

        ###
        # real_Q = Q.reshape(num_nodes, self.feat_dim)
        # real_k = K.reshape(num_nodes, self.feat_dim)
        # real_V = V.reshape(num_nodes, self.feat_dim)

        # A = torch.exp(torch.matmul(real_Q, real_k.transpose(0, 1))
        #     / self.feat_dim ** 0.5)
        # d = torch.matmul(A, torch.ones(num_nodes))
        # D_inv = torch.diag(1 / d)
        # real = torch.matmul(D_inv, torch.matmul(A, real_V))

        # import pdb
        # pdb.set_trace()

        # print(abs(out - real).mean())
        ####

        return out


class InteractionBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 r_cut,
                 gamma,
                 bern_k,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 max_z=DEFAULT_MAX_Z):
        super().__init__()

        self.residual_1 = Residual(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout)
        self.residual_2 = Residual(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout)
        self.res_mlp = ResidualMLP(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout)
        self.local = LocalInteraction(feat_dim=feat_dim,
                                      bern_k=bern_k,
                                      gamma=gamma,
                                      r_cut=r_cut,
                                      activation=activation,
                                      max_z=max_z,
                                      dropout=dropout)
        self.non_local = NonLocalInteraction(feat_dim=feat_dim,
                                             activation=activation,
                                             dropout=dropout)

    def forward(self, x):

        x_tilde = self.residual_1(x)
        l = self.local(x_tilde)
        n = self.non_local(x_tilde)

        x_t = self.residual_2(x_tilde + l + n)
        y_t = self.resmlp(x_t)

        return x_t, y_t


def sigma(x):
    out = torch.where(x > 0,
                      torch.exp(-1 / x),
                      torch.Tensor([0]))
    return out


def f_switch(r, r_on, r_off):
    arg = (r - r_on) / (r_off - r_on)
    num = sigma(1 - arg)
    denom = sigma(1 - arg) + sigma(arg)
    out = num / denom

    return out


class Electrostatics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, xyz):
        n = xyz.shape[0]
        r_ij = (xyz.expand(n, n, 3) -
                xyz.expand(n, n, 3).transpose(0, 1)
                ).pow(2).sum(dim=2).sqrt()

        mask = r_ij > 0
        nbrs = mask.nonzero(as_tuple=False)
        nbrs = nbrs[nbrs[:, 1] > nbrs[:, 0]]

        q_i = q[nbrs[:, 0]]
        q_j = q[nbrs[:, 1]]
        r = r_ij[nbrs[:, 0], nbrs[:, 1]]
        arg_0 = f_switch(r) / (r ** 2 + 1) ** 0.5
        arg_1 = (1 - f_switch(r)) / r
        energy = (KE_KCAL * q_i * q_j * (arg_0 + arg_1)).sum()

        return energy
