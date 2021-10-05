import torch
from torch import nn
from nff.nn.layers import Dense
from nff.utils.scatter import scatter_add
from nff.nn.modules.painn import InvariantMessage, preprocess_r, norm
from nff.nn.modules.spooky import (scatter_pairwise, get_f_switch,
                                   scatter_mol, DEFAULT_MAX_Z, get_dipole,
                                   NuclearEmbedding)
from nff.utils.constants import KE_KCAL, BOHR_RADIUS
from nff.utils.tools import layer_types

DEFAULT_DROPOUT = 0


class NonLocalInteraction(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 nb_features=None,
                 dropout=DEFAULT_DROPOUT):

        from performer_pytorch import FastAttention

        super().__init__()
        self.feat_dim = feat_dim
        if nb_features is None:
            nb_features = feat_dim
        self.attn = FastAttention(dim_heads=feat_dim,
                                  nb_features=nb_features,
                                  causal=False)
        self.dense = Dense(in_features=feat_dim,
                           out_features=(3 * feat_dim),
                           bias=True,
                           dropout_rate=dropout)

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
            # this doesn't work because q and k get
            # multiplied together to give +400
            
            pad = -20 * torch.ones(extra_pad,
                                   self.feat_dim,
                                   device=Q.device)

            q_pad = torch.cat([q, pad])
            k_pad = torch.cat([k, pad])
            v_pad = torch.cat([v, pad])

            q_pads.append(q_pad)
            k_pads.append(k_pad)
            v_pads.append(v_pad)

        q_pads = torch.stack(q_pads)
        k_pads = torch.stack(k_pads)
        v_pads = torch.stack(v_pads)

        return q_pads, k_pads, v_pads

    def forward(self,
                s,
                num_atoms):

        Q, K, V = torch.split(self.dense(s),
                              [self.feat_dim] * 3,
                              dim=-1)

        if not isinstance(num_atoms, list):
            num_atoms = num_atoms.tolist()

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

        return att


class MessageBlock(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout,
                 fast_feats):
        super().__init__()
        self.inv_message = InvariantMessage(feat_dim=feat_dim,
                                            activation=activation,
                                            n_rbf=n_rbf,
                                            cutoff=cutoff,
                                            learnable_k=learnable_k,
                                            dropout=dropout)
        self.nl = NonLocalInteraction(feat_dim=feat_dim,
                                      activation=activation,
                                      dropout=dropout,
                                      nb_features=fast_feats)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                num_atoms,
                **kwargs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_l = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)
        delta_s_nl = self.nl(s=s_j,
                             num_atoms=num_atoms)
        delta_s_i = delta_s_l + delta_s_nl

        return delta_s_i, delta_v_i


class GatedEquivariant(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()

        self.feat_dim = feat_dim
        self.W_1 = Dense(in_features=feat_dim,
                         out_features=feat_dim,
                         bias=False,
                         dropout_rate=dropout)
        self.W_2 = Dense(in_features=feat_dim,
                         out_features=1,
                         bias=False,
                         dropout_rate=dropout)
        self.block = nn.Sequential(Dense(in_features=(2 * feat_dim),
                                         out_features=feat_dim,
                                         bias=True,
                                         dropout_rate=dropout),
                                   layer_types[activation](),
                                   Dense(in_features=feat_dim,
                                         out_features=2,
                                         bias=True,
                                         dropout_rate=dropout)
                                   )

    def forward(self,
                s_i,
                v_i):

        # v has dimension num_nodes x F x 3
        # W has dimension F x F
        # wv has dimension num_nodes x F x 3
        w_v_1 = torch.einsum('ij, kjl -> kil',
                             self.W_1.weight, v_i)
        w_v_2 = torch.einsum('ij, kjl -> kil',
                             self.W_2.weight, v_i)

        cat_s = torch.cat([norm(w_v_1), s_i], dim=-1)
        out = self.block(cat_s)

        split = out[:, :1], out[:, 1:]
        new_v = split[0] * w_v_2.reshape(-1, 3)
        new_s = split[1]

        return new_s, new_v


class GatedInvariant(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()

        self.feat_dim = feat_dim
        self.W_1 = Dense(in_features=feat_dim,
                         out_features=feat_dim,
                         bias=False,
                         dropout_rate=dropout)
        self.block = nn.Sequential(
            Dense(in_features=(2 * feat_dim),
                  out_features=feat_dim,
                  bias=True,
                  dropout_rate=dropout),
            layer_types[activation](),
            Dense(in_features=feat_dim,
                  out_features=1,
                  bias=True,
                  dropout_rate=dropout)
        )

    def forward(self,
                s_i,
                v_i):

        # v has dimension num_nodes x F x 3
        # W has dimension F x F
        # wv has dimension num_nodes x F x 3
        w_v_1 = torch.einsum('ij, kjl -> kil',
                             self.W_1.weight, v_i)

        cat_s = torch.cat([norm(w_v_1), s_i], dim=-1)
        new_s = self.block(cat_s)

        return new_s, None


class Electrostatics(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 r_cut,
                 charge_charge=True,
                 charge_dipole=False,
                 dipole_dipole=False,
                 point_dipoles=False,
                 max_z=DEFAULT_MAX_Z,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()

        gate_class = (GatedEquivariant if point_dipoles
                      else GatedInvariant)
        self.gated = gate_class(feat_dim=feat_dim,
                                activation=activation,
                                dropout=dropout)
        self.r_on = r_cut / 4
        self.r_off = 3 * r_cut / 4
        self.charge_charge = charge_charge
        self.charge_dipole = charge_dipole
        self.dipole_dipole = dipole_dipole
        self.point_dipoles = point_dipoles

    def f_switch(self, r):
        out = get_f_switch(r=r,
                           r_on=self.r_on,
                           r_off=self.r_off)

        return out

    def charge_and_dip(self,
                       xyz,
                       s_i,
                       v_i,
                       z,
                       total_charge,
                       num_atoms):

        atom_charges, atom_dipoles = self.gated(s_i=s_i,
                                                v_i=v_i)
        mol_sum = scatter_mol(atomwise=atom_charges,
                              num_atoms=num_atoms).reshape(-1)
        correction = 1 / num_atoms * (total_charge - mol_sum)
        new_charges = []
        for i, n in enumerate(num_atoms):
            counter = num_atoms[:i].sum()
            old_val = atom_charges[counter: counter + n]
            new_val = old_val + correction[i]
            new_charges.append(new_val)

        atom_charges = torch.cat(new_charges)
        summed_chg_dipole = get_dipole(xyz=xyz,
                                       q=atom_charges,
                                       num_atoms=num_atoms)

        full_dipole = summed_chg_dipole
        if self.point_dipoles:
            summed_atomwise_dipole = scatter_mol(atomwise=atom_dipoles,
                                                 num_atoms=num_atoms)

            full_dipole = full_dipole + summed_atomwise_dipole

        return atom_charges, atom_dipoles, full_dipole

    def get_charge_charge(self,
                          dist,
                          q,
                          num_atoms,
                          mol_nbrs):

        q_i = q[mol_nbrs[:, 0]].reshape(-1)
        q_j = q[mol_nbrs[:, 1]].reshape(-1)

        arg_0 = (self.f_switch(dist)
                 / (dist ** 2 + BOHR_RADIUS ** 2) ** 0.5)
        arg_1 = (1 - self.f_switch(dist)) / dist
        pairwise = (KE_KCAL * q_i * q_j * (arg_0 + arg_1))

        energy = (scatter_pairwise(pairwise=pairwise,
                                   num_atoms=num_atoms,
                                   nbrs=mol_nbrs)
                  .reshape(-1, 1))

        return energy

    def get_charge_dip(self,
                       unit_r_ij,
                       dist,
                       q,
                       dip,
                       num_atoms,
                       mol_nbrs):

        # r_i is the vector pointing to the point charge
        # r_j is the vector pointing to the dipole
        # r_ij = r_i - r_j

        q_i = q[mol_nbrs[:, 0]].reshape(-1, 1)
        dip_j = dip[mol_nbrs[:, 1]]
        # (q_i \vec{r}_ij) \cdot \mu_j / |r_ij|
        numerator = (q_i * unit_r_ij * dip_j).sum(-1)

        arg_0 = (self.f_switch(dist)
                 / (dist ** 2 + BOHR_RADIUS ** 2))
        arg_1 = (1 - self.f_switch(dist)) / dist ** 2

        pairwise = (KE_KCAL * numerator * (arg_0 + arg_1))
        energy = (scatter_pairwise(pairwise=pairwise,
                                   num_atoms=num_atoms,
                                   nbrs=mol_nbrs)
                  .reshape(-1, 1))

        return energy

    def get_dip_dip(self,
                    unit_r_ij,
                    dist,
                    q,
                    dip,
                    num_atoms,
                    mol_nbrs):

        dip_i = dip[mol_nbrs[:, 0]]
        dip_j = dip[mol_nbrs[:, 1]]

        # \vec{\mu}_i \cdot \vec{\mu}_j
        # - (3 (\vec{\mu}_i \cdot \vec_{r}_{ij})
        # \vec{r}_ij \cdot \vec{mu}_j) / |r_ij|^2

        numerator = (
            (dip_i * dip_j).sum(-1)
            - 3 * (unit_r_ij * dip_i).sum(-1)
            * (unit_r_ij * dip_j).sum(-1)
        )

        arg_0 = (self.f_switch(dist)
                 / (dist ** 2 + BOHR_RADIUS ** 2) ** (3 / 2))
        arg_1 = (1 - self.f_switch(dist)) / dist ** 3
        pairwise = (KE_KCAL * numerator * (arg_0 + arg_1))

        energy = (scatter_pairwise(pairwise=pairwise,
                                   num_atoms=num_atoms,
                                   nbrs=mol_nbrs)
                  .reshape(-1, 1))

        return energy

    def get_en(self,
               unit_r_ij,
               dist,
               q,
               dip,
               num_atoms,
               mol_nbrs):

        en = self.get_charge_charge(dist=dist,
                                    q=q,
                                    num_atoms=num_atoms,
                                    mol_nbrs=mol_nbrs)

        if not self.point_dipoles:
            return en

        if self.charge_dipole:
            c_d = self.get_charge_dip(unit_r_ij=unit_r_ij,
                                      dist=dist,
                                      q=q,
                                      dip=dip,
                                      num_atoms=num_atoms,
                                      mol_nbrs=mol_nbrs)

            en = en + c_d

        if self.dipole_dipole:
            d_d = self.get_dip_dip(unit_r_ij=unit_r_ij,
                                   dist=dist,
                                   q=q,
                                   dip=dip,
                                   num_atoms=num_atoms,
                                   mol_nbrs=mol_nbrs)

            en = en + d_d

        return en

    def forward(self,
                s_i,
                v_i,
                z,
                xyz,
                total_charge,
                num_atoms,
                mol_nbrs,
                mol_offsets):

        idx = (mol_nbrs[:, 1] > mol_nbrs[:, 0])
        mol_nbrs = mol_nbrs[idx]
        mol_offsets = mol_offsets[idx]

        q, dip_atom, full_dip = self.charge_and_dip(xyz=xyz,
                                                    s_i=s_i,
                                                    v_i=v_i,
                                                    z=z,
                                                    total_charge=total_charge,
                                                    num_atoms=num_atoms)

        # This is r_ij (=r_i - r_j), not r_ji
        r_ij = xyz[mol_nbrs[:, 0]] - xyz[mol_nbrs[:, 1]] - mol_offsets
        dist = norm(r_ij)
        # unit vector (r_i - r_j) / || r_i - r_j||
        unit_r_ij = r_ij / dist.reshape(-1, 1)
        energy = self.get_en(unit_r_ij=unit_r_ij,
                             dist=dist,
                             q=q,
                             dip=dip_atom,
                             num_atoms=num_atoms,
                             mol_nbrs=mol_nbrs)

        return energy, q, dip_atom, full_dip


class ElectronicEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.linear = Dense(in_features=feat_dim,
                            out_features=feat_dim,
                            bias=True,
                            activation=None,
                            dropout_rate=dropout)
        self.feat_dim = feat_dim
        self.dense = Dense(in_features=feat_dim,
                           out_features=feat_dim,
                           bias=True,
                           activation=layer_types[activation](),
                           dropout_rate=dropout)
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
            this_e_psi = self.dense(av)
            e_psi[counter: counter + num_atoms[j]] = this_e_psi
            counter += num_atoms[j]

        return e_psi


class CombinedEmbedding(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 max_z=DEFAULT_MAX_Z,
                 dropout=DEFAULT_DROPOUT):

        super().__init__()
        self.feat_dim = feat_dim
        self.nuc_embedding = NuclearEmbedding(
            feat_dim=feat_dim,
            max_z=max_z)
        self.charge_embedding = ElectronicEmbedding(
            feat_dim=feat_dim,
            activation=activation,
            dropout=dropout)
        self.spin_embedding = ElectronicEmbedding(
            feat_dim=feat_dim,
            activation=activation,
            dropout=dropout)

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

        s_i = e_z + e_q + e_s
        v_i = (torch.zeros(num_atoms.sum(),
                           self.feat_dim, 3)
               .to(s_i.device))

        return s_i, v_i
