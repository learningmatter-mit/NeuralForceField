import torch
from torch import nn
import sympy as sym

from nff.utils.tools import layer_types
from nff.nn.layers import PreActivation, Dense
# from nff.nn.layers import (PainnRadialBasis, CosineEnvelope,
#                            ExpNormalBasis, Dense)
from nff.utils.scatter import scatter_add
from nff.nn.modules.schnet import ScaleShift, get_act
from nff.utils.constants import ELEC_CONFIG
from nff.utils import make_g_funcs

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
        self.residual = Residual(feat_dim=feat_dim,
                                 activation=activation,
                                 dropout=dropout,
                                 bias=False)
        names = ['k_plus', 'k_minus', 'v_plus', 'v_minus']
        for name in names:
            setattr(self, name, nn.Parameter(torch.ones(feat_dim)
                                             .reshape(-1, 1)))
            nn.init.xavier_uniform_(getattr(self, name))

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
            this_e_psi = self.residual(av)
            e_psi[counter: counter + num_atoms[j]] = this_e_psi
            counter += num_atoms[j]

        return e_psi


class CombinedEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z):

        super().__init__()
        self.nuc_embedding = NuclearEmbedding(feat_dim=feat_dim,
                                              max_z=max_z)
        self.charge_embedding = ElectronicEmbedding(feat_dim=feat_dim,
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
            setattr(self, key, Residual(feat_dim=feat_dim,
                                        activation=activation,
                                        dropout=dropout))

        for key in ["G_s", "G_p", "G_d"]:
            val = nn.Parameter(torch.ones(feat_dim,
                                          bern_k))
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
        residual = res_block(x_j)

        per_nbr = (residual.reshape(n_nbrs, -1, 1)
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

        s_i = quants[0]
        s_term = s_i.reshape(graph_size, -1)

        invariants = []

        for quant, orbital in zip(quants[1:],
                                  orbitals[1:]):
            invariant = self.take_inner(quant,
                                        orbital)
            invariants.append(invariant)

        inp = c_term + s_term + sum(invariants)
        out = self.resmlp_l(inp)

        return out
