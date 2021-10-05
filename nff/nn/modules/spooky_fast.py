import torch
from torch import nn
from nff.nn.layers import (PreActivation, Dense, zeros_initializer,
                           BatchedPreActivation)
from nff.utils.tools import layer_types, make_undirected
from nff.utils.scatter import scatter_add
from nff.utils.constants import ELEC_CONFIG, KE_KCAL, BOHR_RADIUS
from nff.utils import spooky_f_cut, make_y_lm, rho_k

import time

EPS = 1e-15
DEFAULT_ACTIVATION = 'learnable_swish'
DEFAULT_MAX_Z = 86
DEFAULT_DROPOUT = 0
DEFAULT_RES_LAYERS = 2


def timing(func):
    def report_time(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        delta = (end - start) * 1000
        func_name = func.__name__
        func_class = func.__class__.__name__
        print("Took %.2f ms to run %s in %s" % (delta, func_name, func_class))

        return out

    return report_time


def repeat_inp(x, num_out):
    ones = [1] * len(x.shape)
    inp = x.repeat(num_out, *ones)

    # import pdb
    # pdb.set_trace()

    # inp = torch.stack([x.clone() for _ in range(num_out)])

    return inp


def norm(vec):
    """
    For stable norm calculation. PyTorch's implementation
    can be unstable
    """
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


# def scatter_mol(atomwise,
#                 atomwise_mol_list):
#     """
#     Add atomic contributions in a batch to their respective
#     geometries. A simple sum is much faster than doing scatter_add
#     because it takes a very long time to make the indices that
#     map atom index to molecule.
#     """

#     graph_size = atomwise_mol_list[-1] + 1
#     out = scatter_add(src=atomwise,
#                       index=atomwise_mol_list,
#                       dim=0,
#                       dim_size=graph_size)

#     return out


# def scatter_pairwise(pairwise,
#                      nbr_mol_list):
#     """
#     Add pair-wise contributions in a batch to their respective
#     geometries
#     """


#     graph_size = nbr_mol_list[-1] + 1
#     out = scatter_add(src=pairwise,
#                       index=nbr_mol_list,
#                       dim=0,
#                       dim_size=graph_size)

#     return out

def scatter_mol(atomwise,
                num_atoms):
    """
    Add atomic contributions in a batch to their respective
    geometries. A simple sum is much faster than doing scatter_add
    because it takes a very long time to make the indices that 
    map atom index to molecule.
    """

    out = []
    atom_split = torch.split(atomwise, num_atoms.tolist())
    for split in atom_split:
        out.append(split.sum(0))
    out = torch.stack(out)

    return out


def scatter_pairwise(pairwise,
                     num_atoms,
                     nbrs):
    """
    Add pair-wise contributions in a batch to their respective
    geometries 
    """

    # mol_idx = []
    # for i, num in enumerate(num_atoms):
    #     mol_idx += [i] * int(num)

    # mol_idx = torch.LongTensor(mol_idx)
    # nbr_to_mol = []
    # for nbr in nbrs:
    #     nbr_to_mol.append(mol_idx[nbr[0]])
    # nbr_to_mol = torch.LongTensor(nbr_to_mol)

    mol_idx = []
    for i, num in enumerate(num_atoms):
        mol_idx += [i] * int(num)

    mol_idx = (torch.LongTensor(mol_idx)
               .to(pairwise.device))
    nbr_to_mol = mol_idx[nbrs[:, 0]]

    out = scatter_add(src=pairwise,
                      index=nbr_to_mol,
                      dim=0,
                      dim_size=len(num_atoms))

    return out


class Residual(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 num_layers=DEFAULT_RES_LAYERS,
                 bias=True):

        super().__init__()
        block = [
            PreActivation(in_features=feat_dim,
                          out_features=feat_dim,
                          activation=activation,
                          dropout_rate=dropout,
                          bias=bias)
            for _ in range(num_layers - 1)
        ]
        block.append(PreActivation(in_features=feat_dim,
                                   out_features=feat_dim,
                                   activation=activation,
                                   dropout_rate=dropout,
                                   bias=bias,
                                   weight_init=zeros_initializer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = x + self.block(x)
        return output


class ResidualMLP(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 bias=True,
                 residual_layers=DEFAULT_RES_LAYERS):

        super().__init__()
        residual = Residual(feat_dim=feat_dim,
                            activation=activation,
                            dropout=dropout,
                            bias=bias,
                            num_layers=residual_layers)
        self.block = nn.Sequential(residual,
                                   layer_types[activation](),
                                   Dense(in_features=feat_dim,
                                         out_features=feat_dim,
                                         bias=bias))

    def forward(self, x):
        output = self.block(x)
        return output


class BatchedResidual(nn.Module):
    def __init__(self,
                 feat_dim,
                 num_out,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 num_layers=DEFAULT_RES_LAYERS,
                 bias=True):

        super().__init__()
        block = [
            BatchedPreActivation(in_features=feat_dim,
                                 out_features=feat_dim,
                                 num_out=num_out,
                                 activation=activation,
                                 dropout_rate=dropout,
                                 bias=bias)
            for _ in range(num_layers - 1)
        ]
        block.append(BatchedPreActivation(in_features=feat_dim,
                                          out_features=feat_dim,
                                          num_out=num_out,
                                          activation=activation,
                                          dropout_rate=dropout,
                                          bias=bias,
                                          weight_init=zeros_initializer))
        self.block = nn.Sequential(*block)
        self.num_out = num_out

    def forward(self, x):
        output = x + self.block(x)
        return output


class BatchedResidualMLP(nn.Module):
    def __init__(self,
                 feat_dim,
                 num_out,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 bias=True,
                 residual_layers=DEFAULT_RES_LAYERS):

        super().__init__()
        residual = BatchedResidual(feat_dim=feat_dim,
                                   num_out=num_out,
                                   activation=activation,
                                   dropout=dropout,
                                   bias=bias,
                                   num_layers=residual_layers)
        nonlinear = layer_types[activation]()
        linear = BatchedPreActivation(in_features=feat_dim,
                                      out_features=feat_dim,
                                      num_out=num_out,
                                      activation=None,
                                      dropout_rate=dropout,
                                      bias=bias)

        self.block = nn.Sequential(residual,
                                   nonlinear,
                                   linear)
        self.num_out = num_out

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
                           activation=None,
                           weight_init=zeros_initializer)
        self.z_embed = nn.Embedding(max_z, feat_dim, padding_idx=0)

    def forward(self, z):
        d_z = self.elec_config[z.long()].to(z.device)
        tilde_e_z = self.z_embed(z.long())
        e_z = self.m_mat(d_z) + tilde_e_z

        return e_z


class ElectronicEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 residual_layers=DEFAULT_RES_LAYERS):
        super().__init__()
        self.linear = Dense(in_features=feat_dim,
                            out_features=feat_dim,
                            bias=True,
                            activation=None)
        self.feat_dim = feat_dim
        self.resmlp = ResidualMLP(feat_dim=feat_dim,
                                  activation=activation,
                                  dropout=dropout,
                                  bias=False,
                                  residual_layers=residual_layers)
        names = ['k_plus', 'k_minus', 'v_plus', 'v_minus']
        for name in names:
            val = nn.Parameter(torch.zeros(feat_dim, 1,
                                           dtype=torch.float32))
            setattr(self, name, val)

    def forward(self,
                psi,
                e_z,
                num_atoms):

        q = self.linear(e_z)
        split_qs = torch.split(q, num_atoms.tolist())
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
            av = a_i.reshape(-1, 1) * v.reshape(1, -1)
            this_e_psi = self.resmlp(av)
            e_psi[counter: counter + num_atoms[j]] = this_e_psi
            counter += num_atoms[j]

        return e_psi


class CombinedEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z,
                 residual_layers=DEFAULT_RES_LAYERS):

        super().__init__()
        self.nuc_embedding = NuclearEmbedding(
            feat_dim=feat_dim,
            max_z=max_z)
        self.charge_embedding = ElectronicEmbedding(
            feat_dim=feat_dim,
            activation=activation,
            residual_layers=residual_layers)
        self.spin_embedding = ElectronicEmbedding(
            feat_dim=feat_dim,
            activation=activation,
            residual_layers=residual_layers)

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


class GBlock(nn.Module):
    def __init__(self,
                 l,
                 r_cut,
                 bern_k,
                 gamma):
        super().__init__()
        self.l = l
        self.r_cut = r_cut
        self.bern_k = bern_k
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.y_lm_fn = make_y_lm(l)

    def forward(self, r_ij):

        r = norm(r_ij).reshape(-1, 1)
        n_pairs = r_ij.shape[0]
        device = r_ij.device

        m_vals = list(range(-self.l, self.l + 1))
        # is this for-loop slow?
        y = torch.stack([self.y_lm_fn(r_ij, r, self.l, m) for m in
                         m_vals]).transpose(0, 1)
        gamma = self.gamma.clamp(0)
        rho = rho_k(r, self.r_cut, self.bern_k, gamma)
        g = torch.ones(n_pairs,
                       self.bern_k,
                       len(m_vals),
                       device=device)
        g = g * rho.reshape(n_pairs, -1, 1)
        g = g * y.reshape(n_pairs, 1, -1)

        return g


class LocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 bern_k,
                 gamma,
                 r_cut,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z,
                 dropout=DEFAULT_DROPOUT,
                 residual_layers=DEFAULT_RES_LAYERS):

        super().__init__()

        # for letter in ["c", "s", "p", "d", "l"]:
        #     key = f"resmlp_{letter}"
        #     val = ResidualMLP(feat_dim=feat_dim,
        #                       activation=activation,
        #                       dropout=dropout,
        #                       residual_layers=residual_layers)
        #     setattr(self, key, val)

        num_orbs = 3
        self.num_channels = (num_orbs + 1)
        self.resmlp = BatchedResidualMLP(feat_dim=feat_dim,
                                         num_out=self.num_channels,
                                         activation=activation,
                                         dropout=dropout,
                                         residual_layers=residual_layers)
        self.resmlp_l = ResidualMLP(feat_dim=feat_dim,
                                    activation=activation,
                                    dropout=dropout,
                                    residual_layers=residual_layers)

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

        letters = {0: "s", 1: "p", 2: "d"}
        for l, letter in letters.items():
            key = f"g_{letter}"
            g_block = GBlock(l=l,
                             r_cut=r_cut,
                             bern_k=bern_k,
                             gamma=gamma)
            setattr(self, key, g_block)

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
                   # x_j,
                   nbrs,
                   graph_size,
                   orbital,
                   res_out):

        # res_block = getattr(self, f"resmlp_{orbital}")
        n_nbrs = nbrs.shape[0]

        matmul = self.g_matmul(r_ij, orbital)
        # resmlp = res_block(x_j)
        # per_nbr = (resmlp.reshape(n_nbrs, -1, 1)
        #            * matmul)

        per_nbr = (res_out.reshape(n_nbrs, -1, 1)
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

        inp = repeat_inp(x_tilde, self.num_channels)
        # inp = torch.stack([x_tilde for _ in range(self.num_channels)])
        res_outs = self.resmlp(inp)

        # import pdb
        # pdb.set_trace()

        orbitals = ['s', 'p', 'd']
        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]
        graph_size = xyz.shape[0]
        quants = []

        for orbital, res_out in zip(orbitals, res_outs[1:]):
            quant = self.make_quant(r_ij=r_ij,
                                    # x_j=x_j,
                                    nbrs=nbrs,
                                    graph_size=graph_size,
                                    orbital=orbital,
                                    res_out=res_out[nbrs[:, 1]])
            quants.append(quant)

        invariants = []

        for quant, orbital in zip(quants[1:],
                                  orbitals[1:]):
            invariant = self.take_inner(quant,
                                        orbital)
            invariants.append(invariant)

        s_i = quants[0].reshape(graph_size, -1)
        c_i = res_outs[0]
        inp = c_i + s_i + sum(invariants)

        out = self.resmlp_l(inp)

        return out


class NonLocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 residual_layers=DEFAULT_RES_LAYERS):

        from performer_pytorch import FastAttention

        super().__init__()

        # no redraw should happen here - only if
        # you call self attention of cross attention
        # as wrappers
        self.attn = FastAttention(dim_heads=feat_dim,
                                  nb_features=feat_dim,
                                  causal=False)
        self.feat_dim = feat_dim
        self.resmlp = BatchedResidualMLP(feat_dim=feat_dim,
                                         num_out=3,
                                         activation=activation,
                                         dropout=dropout,
                                         residual_layers=residual_layers)

        # for letter in ['q', 'k', 'v']:
        #     key = f'resmlp_{letter}'
        #     val = ResidualMLP(feat_dim=feat_dim,
        #                       activation=activation,
        #                       dropout=dropout,
        #                       residual_layers=residual_layers)
        #     setattr(self, key, val)

    def forward(self,
                x_tilde,
                num_atoms):

        # x_tilde has dimension N x F
        # N = number of nodes, F = feature dimension

        num_nodes = x_tilde.shape[0]
        # Q = self.resmlp_q(x_tilde)
        # K = self.resmlp_k(x_tilde)
        # V = self.resmlp_v(x_tilde)

        inp = repeat_inp(x_tilde, 3)
        Q, K, V = self.resmlp(inp)

        q_split = torch.split(Q, num_atoms.tolist())
        k_split = torch.split(K, num_atoms.tolist())
        v_split = torch.split(V, num_atoms.tolist())

        # max_atoms = num_atoms.max()
        # q_stack = []
        # k_stack = []
        # v_stack = []

        # for i, num in enumerate(num_atom)s:
        #   delta = max_atoms - num
        #   pad = torch.zeros(delta, self.feat_dim)

        #   q_stack.append(torch.cat(q_split[i], pad))
        #   k_stack.append(torch.cat(k_split[i], pad))
        #   v_stack.append(torch.cat(v_split[i], pad))

        # q_stack = torch.stack(q_stack)
        # k_stack = torch.stack(k_stack)
        # v_stack = torch.stack(v_stack)

        out = torch.zeros(num_nodes,
                          self.feat_dim,
                          device=x_tilde.device)
        for i, num in enumerate(num_atoms):
            q = q_split[i].reshape(1, 1, -1, self.feat_dim)
            k = k_split[i].reshape(1, 1, -1, self.feat_dim)
            v = v_split[i].reshape(1, 1, -1, self.feat_dim)
            att = (self.attn(q, k, v)
                   .reshape(-1, self.feat_dim))

            ###
            # real_q = q.reshape(-1, self.feat_dim)
            # real_k = k.reshape(-1, self.feat_dim)
            # real_v = v.reshape(-1, self.feat_dim)

            # A = torch.exp(torch.matmul(real_q, real_k.transpose(0, 1))
            #               / self.feat_dim ** 0.5)
            # ones = torch.ones(real_q.shape[0],
            #                   dtype=torch.float32,
            #                   device=real_q.device)
            # D_inv = torch.diag(1 / torch.matmul(A, ones))
            # real_att = torch.matmul(D_inv, torch.matmul(A, real_v))

            # att = real_att
            # err = abs(real_att - att).mean()
            # abs_mean = att.abs().mean()

            # import pdb
            # pdb.set_trace()

            # print(err)
            # print(abs_mean)
            # print(err / abs_mean * 100)
            ###

            counter = sum(num_atoms[:i])
            out[counter: counter + num] = att

        import pdb
        # pdb.set_trace()

        return out


class InteractionBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 r_cut,
                 gamma,
                 bern_k,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 max_z=DEFAULT_MAX_Z,
                 residual_layers=DEFAULT_RES_LAYERS):
        super().__init__()

        self.residual_1 = Residual(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout,
                                   num_layers=residual_layers)
        self.residual_2 = Residual(feat_dim=feat_dim,
                                   activation=activation,
                                   dropout=dropout,
                                   num_layers=residual_layers)
        self.resmlp = ResidualMLP(feat_dim=feat_dim,
                                  activation=activation,
                                  dropout=dropout,
                                  residual_layers=residual_layers)
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

    def forward(self,
                x,
                xyz,
                nbrs,
                num_atoms):

        x_tilde = self.residual_1(x)
        l = self.local(xyz=xyz,
                       x_tilde=x_tilde,
                       nbrs=nbrs)
        n = self.non_local(x_tilde=x_tilde,
                           num_atoms=num_atoms)
        x_t = self.residual_2(x_tilde + l + n)
        # x_t = self.residual_2(x_tilde + l )

        y_t = self.resmlp(x_t)

        return x_t, y_t


def get_f_switch(r, r_on, r_off):

    arg = (r - r_on) / (r_off - r_on)
    x = arg
    y = 1 - arg
    exp_arg = (x - y) / (x * y)

    zero = torch.Tensor([0]).to(r.device)
    one = torch.Tensor([1.0]).to(r.device)
    out = torch.ones_like(x)

    mask = (x > 0) * (y > 0)
    # For numerical stability
    # Anything > 34 will give under 1e-15
    mask_zero = mask * (exp_arg >= 34)
    mask_nonzero = mask * (exp_arg < 34)

    out[mask_nonzero] = 1 / (1 + torch.exp(exp_arg[mask_nonzero]))
    out[mask_zero] = zero

    out[(x > 0) * (y <= 0)] = zero
    out[(x <= 0) * (y > 0)] = one
    out[(x <= 0) * (y <= 0)] = zero

    return out


class Electrostatics(nn.Module):
    def __init__(self,
                 feat_dim,
                 r_cut,
                 max_z=DEFAULT_MAX_Z):
        super().__init__()

        self.w = Dense(in_features=feat_dim,
                       out_features=1,
                       bias=False,
                       activation=None)
        self.z_embed = nn.Embedding(max_z, 1, padding_idx=0)
        self.r_on = r_cut / 4
        self.r_off = 3 * r_cut / 4

    def f_switch(self, r):
        out = get_f_switch(r=r,
                           r_on=self.r_on,
                           r_off=self.r_off)

        return out

    def get_charge(self,
                   f,
                   z,
                   total_charge,
                   num_atoms):

        w_f = self.w(f)
        q_z = self.z_embed(z)
        charge = w_f + q_z
        mol_sum = scatter_mol(atomwise=charge,
                              num_atoms=num_atoms).reshape(-1)
        correction = 1 / num_atoms * (total_charge - mol_sum)
        new_charges = []
        for i, n in enumerate(num_atoms):
            counter = num_atoms[:i].sum()
            old_val = charge[counter: counter + n]
            new_val = old_val + correction[i]
            new_charges.append(new_val)

        new_charges = torch.cat(new_charges)

        return new_charges

    def get_en(self,
               q,
               xyz,
               num_atoms,
               mol_nbrs):

        r_ij = norm(xyz[mol_nbrs[:, 0]] - xyz[mol_nbrs[:, 1]])
        q_i = q[mol_nbrs[:, 0]].reshape(-1)
        q_j = q[mol_nbrs[:, 1]].reshape(-1)

        arg_0 = (self.f_switch(r_ij)
                 / (r_ij ** 2 + BOHR_RADIUS ** 2) ** 0.5)
        arg_1 = (1 - self.f_switch(r_ij)) / r_ij
        pairwise = (KE_KCAL * q_i * q_j * (arg_0 + arg_1))

        energy = (scatter_pairwise(pairwise=pairwise,
                                   num_atoms=num_atoms,
                                   nbrs=mol_nbrs)
                  .reshape(-1, 1))

        return energy

    def forward(self,
                f,
                z,
                xyz,
                total_charge,
                num_atoms,
                mol_nbrs):

        q = self.get_charge(f=f,
                            z=z,
                            total_charge=total_charge,
                            num_atoms=num_atoms)
        energy = self.get_en(q=q,
                             xyz=xyz,
                             num_atoms=num_atoms,
                             mol_nbrs=mol_nbrs)

        return energy, q


class NuclearRepulsion(nn.Module):
    def __init__(self, r_cut):
        super().__init__()
        self.r_cut = r_cut
        self.d = torch.Tensor([0.8853 * BOHR_RADIUS])
        self.z_exp = torch.Tensor([0.23])
        self.c = torch.Tensor([0.1818, 0.5099, 0.2802, 0.02817])
        self.exponents = torch.Tensor([3.2, 0.9423, 0.4029, 0.2016])

        for key in ["d", "z_exp", "c", "exponents"]:
            val = getattr(self, key)
            new_val = nn.Parameter(val.reshape(-1, 1))
            setattr(self, key, new_val)

    def zbl_phi(self,
                r_ij,
                z_i,
                z_j):

        d = self.d.clamp(0)
        z_exp = self.z_exp.clamp(0)
        c = self.c.clamp(0)
        c = c / c.sum()
        exponents = self.exponents.clamp(0)

        a = ((d / (z_i ** z_exp + z_j ** z_exp))
             .reshape(-1))

        out = (c * torch.exp(-exponents * r_ij.reshape(-1) / a)
               ).sum(0)

        return out

    def forward(self,
                xyz,
                z,
                nbrs,
                num_atoms):

        undirec = make_undirected(nbrs)
        z_i = z[undirec[:, 0]].to(torch.float32)
        z_j = z[undirec[:, 1]].to(torch.float32)
        r_ij = norm(xyz[undirec[:, 0]] - xyz[undirec[:, 1]])

        phi = self.zbl_phi(r_ij=r_ij,
                           z_i=z_i,
                           z_j=z_j)
        pairwise = (KE_KCAL * z_i * z_j / r_ij
                    * phi
                    * spooky_f_cut(r_ij, self.r_cut))
        energy = scatter_pairwise(pairwise=pairwise,
                                  num_atoms=num_atoms,
                                  nbrs=undirec).reshape(-1, 1)

        # this_z_i = z[undirec[:, 0][:36]]
        # this_z_j = z[undirec[:, 1][:36]]
        # this_rij = ((xyz[undirec[:, 0]] - xyz[undirec[:, 1]]
        #              ) ** 2).sum(-1) ** 0.5
        # test = KE_KCAL * this_z_i * this_z_j / this_rij[:36]
        # test *= spooky_f_cut(this_rij[:36], self.r_cut)

        # c_norm = self.c / self.c.sum()
        # test *= (
        #     (c_norm[0] * torch.exp(-this_rij[:36] / self.d[0]
        #                            * self.exponents[0]
        #                            * (this_z_i ** self.z_exp
        #                               + this_z_j ** self.z_exp)))
        #     + (c_norm[1] * torch.exp(-this_rij[:36] / self.d[0]
        #                              * self.exponents[1]
        #                              * (this_z_i ** self.z_exp
        #                                 + this_z_j ** self.z_exp)))
        #     + (c_norm[2] * torch.exp(-this_rij[:36] / self.d[0]
        #                              * self.exponents[2]
        #                              * (this_z_i ** self.z_exp
        #                                 + this_z_j ** self.z_exp)))
        #     + (c_norm[3] * torch.exp(-this_rij[:36] / self.d[0]
        #                              * self.exponents[3]
        #                              * (this_z_i ** self.z_exp
        #                                 + this_z_j ** self.z_exp)))
        # ).reshape(-1)

        # import pdb
        # pdb.set_trace()

        # test = test.sum()

        # print(test)
        # print(energy[0])

        return energy


class AtomwiseReadout(nn.Module):
    def __init__(self,
                 feat_dim,
                 max_z=DEFAULT_MAX_Z):
        super().__init__()
        self.w_e = Dense(in_features=feat_dim,
                         out_features=1,
                         bias=False,
                         activation=None)
        self.z_bias = nn.Embedding(max_z, 1, padding_idx=0)

    def forward(self,
                z,
                f,
                num_atoms):

        atomwise = self.w_e(f) + self.z_bias(z)
        e_total = scatter_mol(atomwise=atomwise,
                              num_atoms=num_atoms)

        return e_total


def get_dipole(xyz,
               q,
               num_atoms):

    qr = q * xyz
    dipole = scatter_mol(atomwise=qr,
                         num_atoms=num_atoms)

    return dipole
