import torch
from torch import nn
from torch.nn.functional import softplus
from nff.nn.layers import PreActivation, Dense, zeros_initializer
from nff.utils.tools import layer_types
from nff.utils.scatter import scatter_add
from nff.utils.constants import ELEC_CONFIG, KE_KCAL, BOHR_RADIUS
from nff.utils import spooky_f_cut, make_y_lm, rho_k

import time

EPS = 1e-15
DEFAULT_ACTIVATION = 'learnable_swish'
DEFAULT_MAX_Z = 86
DEFAULT_DROPOUT = 0
DEFAULT_RES_LAYERS = 2


ZBL = {"d": [0.8854 * BOHR_RADIUS],
       "z_exp": [0.23],
       "c": [0.1818, 0.5099, 0.2802, 0.02817],
       "exponents": [3.2, 0.9423, 0.4028, 0.2016]}


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
        dense = Dense(in_features=feat_dim,
                      out_features=feat_dim,
                      bias=bias)
        self.block = nn.Sequential(residual,
                                   layer_types[activation](),
                                   dense)

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
                           activation=None,
                           weight_init=zeros_initializer)
        self.z_embed = nn.Embedding(max_z, feat_dim, padding_idx=0)

    def forward(self, z):
        d_z = self.elec_config[z.long()].to(z.device)
        if torch.isnan(d_z).any():
            z_list = z.detach().long().tolist()
            unique_z = torch.LongTensor(list(set(z_list)))
            nan_idx = self.elec_config[unique_z].isnan()[:, 0]
            missing = unique_z[nan_idx].reshape(-1).tolist()
            msg = f"Missing elements {missing} from elec_config.json"
            raise Exception(msg)

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
                   / self.feat_dim ** 0.5).reshape(-1, 1)
            zero = torch.zeros_like(arg)
            num = torch.logsumexp(
                torch.cat([zero, arg], dim=-1),
                dim=1
            )
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
        e_q = self.charge_embedding(psi=charge.reshape(-1),
                                    e_z=e_z,
                                    num_atoms=num_atoms)
        e_s = self.spin_embedding(psi=spin.reshape(-1),
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
        self.gamma_inv = nn.Parameter(
            torch.log(
                torch.exp(torch.Tensor([gamma])) - 1
            )
        )
        self.y_lm_fn = make_y_lm(l)

    @property
    def gamma(self):
        return softplus(self.gamma_inv)

    def forward(self,
                r_ij,
                r):

        n_pairs = r_ij.shape[0]
        device = r_ij.device

        m_vals = list(range(-self.l, self.l + 1))
        # is this for-loop slow?
        y = torch.stack([self.y_lm_fn(r_ij, r, self.l, m) for m in
                         m_vals]).transpose(0, 1)

        rho = rho_k(r, self.r_cut, self.bern_k, self.gamma)
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
                 l_max=2,
                 activation=DEFAULT_ACTIVATION,
                 max_z=DEFAULT_MAX_Z,
                 dropout=DEFAULT_DROPOUT,
                 residual_layers=DEFAULT_RES_LAYERS):

        super().__init__()

        self.l_vals = list(range(l_max + 1))
        for suffix in ["c", "l", *self.l_vals]:
            key = f"resmlp_{suffix}"
            val = ResidualMLP(feat_dim=feat_dim,
                              activation=activation,
                              dropout=dropout,
                              residual_layers=residual_layers)
            setattr(self, key, val)

        for l in self.l_vals:
            key = f"G_{l}"
            val = nn.Parameter(torch.ones(feat_dim,
                                          bern_k))
            nn.init.xavier_uniform_(val)
            setattr(self, key, val)

        for l in self.l_vals[1:]:
            # Called P_1, P_2, D_1, D_2 in original paper
            keys = [f"mix_mat_{l}_1", f"mix_mat_{l}_2"]
            for key in keys:
                val = nn.Parameter(torch.ones(feat_dim,
                                              feat_dim))
                nn.init.xavier_uniform_(val)
                setattr(self, key, val)

        for l in self.l_vals:
            key = f"g_{l}"
            g_block = GBlock(l=l,
                             r_cut=r_cut,
                             bern_k=bern_k,
                             gamma=gamma)
            setattr(self, key, g_block)

    def g_matmul(self,
                 r_ij,
                 l,
                 r):

        g_func = getattr(self, f"g_{l}")
        g = g_func(r_ij, r)
        G = getattr(self, f"G_{l}")

        # g: N_nbrs x K x (1, 3, or 5)
        # G: F x K
        # output: N_nbrs x F x (1, 3, or 5)

        out = torch.einsum('ik, jkl -> jil', G, g)
        return out

    def make_quant(self,
                   r_ij,
                   x_tilde,
                   nbrs,
                   graph_size,
                   l,
                   r):

        res_block = getattr(self, f"resmlp_{l}")
        n_nbrs = nbrs.shape[0]
        matmul = self.g_matmul(r_ij=r_ij,
                               l=l,
                               r=r)
        resmlp = res_block(x_tilde)[nbrs[:, 1]]
        per_nbr = (resmlp.reshape(n_nbrs, -1, 1)
                   * matmul)
        out = scatter_add(src=per_nbr,
                          index=nbrs[:, 0],
                          dim=0,
                          dim_size=graph_size)

        return out

    def take_inner(self,
                   quant,
                   l):

        name = f"mix_mat_{l}"

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
                nbrs,
                r_ij):

        # dimension N_nbrs x F
        graph_size = xyz.shape[0]
        c_term = self.resmlp_c(x_tilde)
        quants = []

        r = norm(r_ij).reshape(-1, 1)
        for l in self.l_vals:
            quant = self.make_quant(r_ij=r_ij,
                                    r=r,
                                    x_tilde=x_tilde,
                                    nbrs=nbrs,
                                    graph_size=graph_size,
                                    l=l)
            quants.append(quant)

        invariants = []

        for quant, l in zip(quants[1:],
                            self.l_vals[1:]):
            invariant = self.take_inner(quant,
                                        l)
            invariants.append(invariant)

        s_i = quants[0]
        s_term = s_i.reshape(graph_size, -1)
        inp = c_term + s_term + sum(invariants)

        out = self.resmlp_l(inp)

        return out


class NonLocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 nb_features=None,
                 activation=DEFAULT_ACTIVATION,
                 dropout=DEFAULT_DROPOUT,
                 residual_layers=DEFAULT_RES_LAYERS):

        from performer_pytorch import FastAttention

        super().__init__()

        # no redraw should happen here - only if
        # you call self attention of cross attention
        # as wrappers

        if nb_features is None:
            nb_features = feat_dim
        self.attn = FastAttention(dim_heads=feat_dim,
                                  nb_features=nb_features,
                                  causal=False)
        self.feat_dim = feat_dim

        for letter in ['q', 'k', 'v']:
            key = f'resmlp_{letter}'
            val = ResidualMLP(feat_dim=feat_dim,
                              activation=activation,
                              dropout=dropout,
                              residual_layers=residual_layers)
            setattr(self, key, val)

    def pad(self,
            Q,
            K,
            V,
            num_atoms):

        q_split = torch.split(Q, num_atoms)
        k_split = torch.split(K, num_atoms)
        v_split = torch.split(V, num_atoms)

        q_pads = []
        k_pads = []
        v_pads = []

        max_num_atoms = int(max(num_atoms))
        for i, q in enumerate(q_split):
            k = k_split[i]
            v = v_split[i]
            extra_pad = max_num_atoms - q.shape[0]
            zeros = torch.zeros(extra_pad,
                                self.feat_dim,
                                device=Q.device)

            q_pad = torch.cat([q, zeros])
            k_pad = torch.cat([k, zeros])
            v_pad = torch.cat([v, zeros])

            q_pads.append(q_pad)
            k_pads.append(k_pad)
            v_pads.append(v_pad)

        q_pads = torch.stack(q_pads)
        k_pads = torch.stack(k_pads)
        v_pads = torch.stack(v_pads)

        return q_pads, k_pads, v_pads

    def forward(self,
                x_tilde,
                num_atoms):

        # x_tilde has dimension N x F
        # N = number of nodes, F = feature dimension

        Q = self.resmlp_q(x_tilde)
        K = self.resmlp_k(x_tilde)
        V = self.resmlp_v(x_tilde)

        if not isinstance(num_atoms, list):
            num_atoms = num_atoms.tolist()

        # if len(list(set(num_atoms))) == 1:
        #     pads = [torch.stack(torch.split(i, num_atoms))
        #             for i in [Q, K, V]]
        #     q_pad, k_pad, v_pad = pads
        # else:
        #     q_pad, k_pad, v_pad = self.pad(Q=Q,
        #                                    K=K,
        #                                    V=V,
        #                                    num_atoms=num_atoms)

        # att = self.attn(q_pad.unsqueeze(0),
        #                 k_pad.unsqueeze(0),
        #                 v_pad.unsqueeze(0)
        #                 ).squeeze(0)

        # att = torch.cat([i[:n] for i, n in zip(att, num_atoms)])

        q_split = torch.split(Q, num_atoms)
        k_split = torch.split(K, num_atoms)
        v_split = torch.split(V, num_atoms)
        att = []

        for i, q in enumerate(q_split):
            k = k_split[i]
            v = v_split[i]

            qk = torch.matmul(q, k.transpose(0, 1)) / self.feat_dim ** 0.5
            this_att = torch.matmul(torch.softmax(qk, dim=-1), v)
            att.append(this_att)

        att = torch.cat(att)

        # # print(abs(base_att - att).mean() / abs(att).mean() * 100)

        # for i, q in enumerate(q_split):
        #     q = q.unsqueeze(0).unsqueeze(0)
        #     k = k_split[i].unsqueeze(0).unsqueeze(0)
        #     v = v_split[i].unsqueeze(0).unsqueeze(0)
        #     this_att = self.attn(q, k, v).reshape(-1, self.feat_dim)
        #     att.append(this_att)

        # att = torch.cat(att)

        return att


class InteractionBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 r_cut,
                 gamma,
                 bern_k,
                 l_max=2,
                 fast_feats=None,
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
                                      dropout=dropout,
                                      l_max=l_max)
        self.non_local = NonLocalInteraction(feat_dim=feat_dim,
                                             activation=activation,
                                             dropout=dropout,
                                             nb_features=fast_feats)

    def forward(self,
                x,
                xyz,
                nbrs,
                num_atoms,
                r_ij):

        x_tilde = self.residual_1(x)
        l = self.local(xyz=xyz,
                       x_tilde=x_tilde,
                       nbrs=nbrs,
                       r_ij=r_ij)
        n = self.non_local(x_tilde=x_tilde,
                           num_atoms=num_atoms)
        x_t = self.residual_2(x_tilde + l + n)
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
    # Anything > 20 will give under 1e-9
    mask_zero = mask * (exp_arg >= 20)
    mask_nonzero = mask * (exp_arg < 20)

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
               mol_nbrs,
               mol_offsets):

        r_ij = norm(xyz[mol_nbrs[:, 0]] - xyz[mol_nbrs[:, 1]]
                    - mol_offsets)
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
                mol_nbrs,
                mol_offsets):

        idx = (mol_nbrs[:, 1] > mol_nbrs[:, 0])
        mol_nbrs = mol_nbrs[idx]
        mol_offsets = mol_offsets[idx]

        q = self.get_charge(f=f,
                            z=z,
                            total_charge=total_charge,
                            num_atoms=num_atoms)
        energy = self.get_en(q=q,
                             xyz=xyz,
                             num_atoms=num_atoms,
                             mol_nbrs=mol_nbrs,
                             mol_offsets=mol_offsets)

        return energy, q


class NuclearRepulsion(nn.Module):
    def __init__(self, r_cut):
        super().__init__()
        self.r_cut = r_cut
        for key, val in ZBL.items():
            # compute inverse softplus
            val = torch.Tensor(val)
            inv_val = nn.Parameter(torch.log(torch.exp(val) - 1)
                                   .reshape(-1, 1))
            setattr(self, key + "_inv", inv_val)

    @property
    def d(self):
        return softplus(self.d_inv)

    @property
    def z_exp(self):
        return softplus(self.z_exp_inv)

    @property
    def exponents(self):
        return softplus(self.exponents_inv)

    @property
    def c(self):
        return softplus(self.c_inv)

    def zbl_phi(self,
                r_ij,
                z_i,
                z_j):

        a = ((self.d / (z_i ** self.z_exp + z_j ** self.z_exp))
             .reshape(-1))

        out = (self.c * torch.exp(-self.exponents * r_ij.reshape(-1) / a)
               ).sum(0) / self.c.sum()

        return out

    def forward(self,
                xyz,
                z,
                nbrs,
                num_atoms,
                offsets):

        idx = (nbrs[:, 1] > nbrs[:, 0])
        undirec = nbrs[idx]
        undirec_offsets = offsets[idx]

        z_i = z[undirec[:, 0]].to(torch.float32)
        z_j = z[undirec[:, 1]].to(torch.float32)

        r_ij = norm(xyz[undirec[:, 0]] - xyz[undirec[:, 1]]
                    - undirec_offsets)

        phi = self.zbl_phi(r_ij=r_ij,
                           z_i=z_i,
                           z_j=z_j)
        pairwise = (
            KE_KCAL
            * z_i * z_j / r_ij
            * phi
            * spooky_f_cut(r_ij, self.r_cut)
        )
        energy = scatter_pairwise(pairwise=pairwise,
                                  num_atoms=num_atoms,
                                  nbrs=undirec).reshape(-1, 1)

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
