import numpy as np

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import MessagePassingModule, EdgeUpdateModule, \
    GeometricOperations, TopologyOperations
from nff.nn.utils import construct_sequential, construct_module_dict
from nff.utils.scatter import compute_grad

import unittest
import itertools
import copy
import pdb

EPSILON = 1e-15
# TOPS = ['bond', 'angle', 'dihedral', 'improper', 'pair']
TOPS = ['bond', 'angle', 'dihedral', 'improper']


class ZeroNet(torch.nn.Module):
    """
    Network to return an array of all zeroes.
    """
    def __init__(self, L_out):
        """
        Args:
            L_out: dimension of zero vector
        """
        super(ZeroNet, self).__init__()
        self.L_out = L_out

    def forward(self, x):
        # written in this roundabout way to ensure that if x is on a GPU
        # then the output will also be on a GPU
        result_layer  = torch.stack([(x.reshape(-1)*0.0)[0].detach()]*self.L_out)
        # one array of zeros per input:
        output = torch.stack([result_layer for _ in range(len(x))])
        return output

class ParameterPredictor(torch.nn.Module):
    """
    Class for predicting parameters from a neural net. Used for AuTopology.
    """
    def __init__(self, L_in, L_hidden, L_out, trainable=False):

        """
        Args:
            L_in: 
            L_hidden:
            L_out:
            trainable: If true, then these parameters have a trainable
                prediction. Otherwise the parameters just return zero.

        """

        super(ParameterPredictor, self).__init__()
        if trainable:
            modules = torch.nn.ModuleList()
            Lh = [L_in] + L_hidden
            for Lh_in, Lh_out in [Lh[i:i + 2] for i in range(len(Lh) - 1)]:
                modules.append(torch.nn.Linear(Lh_in, Lh_out))
                modules.append(torch.nn.Tanh())
            modules.append(torch.nn.Linear(Lh[-1], L_out))
            self.net = torch.nn.Sequential(*modules)
        else:
            self.net = ZeroNet(L_out)

    def forward(self, x):
        return self.net(x)


class BondNet(torch.nn.Module):
    def __init__(self, Fr, Lh, terms=['harmonic'], trainable=False):
        super(BondNet, self).__init__()
        self.Fr = Fr
        self.Lh = Lh
        self.terms = terms
        self.true_params = None
        self.learned_params = {}

        if 'harmonic' in self.terms:
            self.r0_harmonic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.k_harmonic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.learned_params['harmonic'] = {'r0': None, 'k': None}
        if 'morse' in self.terms:

            self.r0_morse = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.a_morse = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.De_morse = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.learned_params['morse'] = {'r0': None, 'a': None, 'De': None}
        if 'cubic' in self.terms:
            self.r0_cubic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.k_cubic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.learned_params['cubic'] = {'r0': None, 'k': None}
        if 'quartic' in self.terms:
            self.r0_quartic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.k_quartic = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.learned_params['quartic'] = {'r0': None, 'k': None}

    def forward(self, r, batch, xyz):

        bonds = batch["bonds"]
        num_bonds = batch["num_bonds"].tolist()

        N = xyz.shape[0]
        D = xyz.expand(N, N, 3) - xyz.expand(N, N, 3).transpose(0, 1)
        D = D.pow(2).sum(dim=2)
        D = D[bonds[:, 0], bonds[:, 1]].pow(0.5).view(-1, 1)
        node_input = r[bonds].sum(1)
        E = 0.0
        if 'harmonic' in self.terms:

            r0_harmonic = ((1.5 ** 0.5) + 0.1 * self.r0_harmonic(node_input)).pow(2)
            k_harmonic = ((100 ** 0.5) + self.k_harmonic(node_input)).pow(2)
            E = E + (k_harmonic / 2) * (D - r0_harmonic).pow(2)
            self.learned_params['harmonic']['r0'] = r0_harmonic.tolist()
            self.learned_params['harmonic']['k'] = k_harmonic.tolist()
        if 'morse' in self.terms:
            self.learned_params['morse'] = {}
            r0_morse = self.r0_morse(node_input).pow(2)
            a_morse = self.a_morse(node_input).pow(2)
            De_morse = self.De_morse(node_input).pow(2)
            E = E + De_morse * (1 - torch.exp(-a_morse * (D - r0_morse))).pow(2)
            self.learned_params['morse']['r0'] = r0_morse.tolist()
            self.learned_params['morse']['a'] = a_morse.tolist()
            self.learned_params['morse']['De'] = De_morse.tolist()
        if 'cubic' in self.terms:
            self.learned_params['cubic'] = {}
            r0_cubic = self.r0_cubic(node_input).pow(2)
            k_cubic = self.k_cubic(node_input).pow(2)
            E = E + (k_cubic / 2) * (D - r0_cubic).pow(3)
            self.learned_params['cubic']['r0'] = r0_cubic.tolist()
            self.learned_params['cubic']['k'] = k_cubic.tolist()
        if 'quartic' in self.terms:
            self.learned_params['quartic'] = {}
            r0_quartic = self.r0_quartic(node_input).pow(2)
            k_quartic = self.k_quartic(node_input).pow(2)
            E = E + (k_quartic / 2) * (D - r0_quartic).pow(4)
            self.learned_params['quartic']['r0'] = r0_quartic.tolist()
            self.learned_params['quartic']['r0'] = k_quartic.tolist()

        # split the results into per-molecule energies through batch["num_bonds"]
        E = torch.stack([e.sum(0) for e in torch.split(E, num_bonds)])
        return (E)


class AngleNet(torch.nn.Module):
    def __init__(self, Fr, Lh, terms=['harmonic'], trainable=False):
        super(AngleNet, self).__init__()
        self.Fr = Fr
        self.Lh = Lh
        self.terms = terms
        self.true_params = None
        self.learned_params = {}
        if 'harmonic' in self.terms:
            self.theta0_harmonic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.k_harmonic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.learned_params['harmonic'] = {'theta0': None, 'k': None}
        if 'cubic' in self.terms:
            self.theta0_cubic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.k_cubic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.learned_params['cubic'] = {'theta0': None, 'k': None}
        if 'quartic' in self.terms:
            self.theta0_quartic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.k_quartic = ParameterPredictor(2 * Fr, Lh, 1, trainable=trainable)
            self.learned_params['quartic'] = {'theta0': None, 'k': None}

    def forward(self, r, batch, xyz):

        angles = batch["angles"]
        num_angles = batch["num_angles"].tolist()

        if num_angles == [0]*len(num_angles):
            return torch.tensor([0.0 for _ in range(len(num_angles))])
            
        N = xyz.shape[0]
        D = xyz.expand(N, N, 3) - xyz.expand(N, N, 3).transpose(0, 1)
        angle_vec1 = D[angles[:, 0], angles[:, 1], :]
        angle_vec2 = D[angles[:, 1], angles[:, 2], :]
        dot_unnorm = (-angle_vec1 * angle_vec2).sum(1)
        norm = torch.sqrt((angle_vec1.pow(2)).sum(1) * (angle_vec2.pow(2)).sum(1))
        cos_theta = (dot_unnorm / norm).view(-1, 1)
        theta = torch.acos(cos_theta / 1.000001)
        node_input = torch.cat([r[angles[:, [0, 2]]].sum(1), r[angles[:, 1]]], dim=1)
        E = 0.0
        if 'harmonic' in self.terms:
            theta0_harmonic = (((109.5 * np.pi / 180) ** 0.5) + self.theta0_harmonic(node_input)).pow(2)
            k_harmonic = ((10 ** 0.5) + self.k_harmonic(node_input)).pow(2)
            E = E + (k_harmonic / 2) * (theta - theta0_harmonic).pow(2)
            self.learned_params['harmonic']['theta0'] = theta0_harmonic.tolist()
            self.learned_params['harmonic']['k'] = k_harmonic.tolist()
        if 'cubic' in self.terms:
            theta0_cubic = self.theta0_cubic(node_input).pow(2)
            k_cubic = self.k_cubic(node_input).pow(2)
            E = E + (k_cubic / 2) * (theta - theta0_cubic).pow(3)
            self.learned_params['cubic']['theta0'] = theta0_cubic.tolist()
            self.learned_params['cubic']['k'] = k_cubic.tolist()
        if 'quartic' in self.terms:
            theta0_quartic = self.theta0_quartic(node_input).pow(2)
            k_quartic = self.k_quartic(node_input).pow(2)
            E = E + (k_quartic / 2) * (theta - theta0_quartic).pow(4)
            self.learned_params['quartic']['theta0'] = theta0_quartic.tolist()
            self.learned_params['quartic']['k'] = k_quartic.tolist()

        # split the results into per-molecule energies through batch["num_angles"]
        E = torch.stack([e.sum(0) for e in torch.split(E, num_angles)])
        return (E)


class DihedralNet(torch.nn.Module):
    def __init__(self, Fr, Lh, terms=['OPLS'], trainable=False):
        super(DihedralNet, self).__init__()
        self.Fr = Fr
        self.Lh = Lh
        self.terms = terms
        self.true_params = None
        self.learned_params = {}
        self.nonlinear = ParameterPredictor(2 * self.Fr, Lh, Lh[-1], trainable=trainable)
        if 'multiharmonic' in self.terms:
            self.dihedralnet_multiharmonic = ParameterPredictor(Lh[-1], Lh, 5, trainable=trainable)
            self.learned_params['multiharmonic'] = {'dihedralnet': None}
        if 'OPLS' in self.terms:
            self.dihedralnet_OPLS = ParameterPredictor(Lh[-1], Lh, 4, trainable=trainable)
            self.learned_params['OPLS'] = {'dihedralnet': None}

    def forward(self, r, batch, xyz):

        dihedrals = batch["dihedrals"]
        num_dihedrals = batch["num_dihedrals"].tolist()

        if num_dihedrals == [0]*len(num_dihedrals):
            return torch.tensor([0.0 for _ in range(len(num_dihedrals))])

        N = xyz.shape[0]
        D = xyz.expand(N, N, 3) - xyz.expand(N, N, 3).transpose(0, 1)
        vec1 = D[dihedrals[:, 1], dihedrals[:, 0]]
        vec2 = D[dihedrals[:, 1], dihedrals[:, 2]]
        vec3 = D[dihedrals[:, 2], dihedrals[:, 1]]
        vec4 = D[dihedrals[:, 2], dihedrals[:, 3]]
        cross1 = torch.cross(vec1, vec2)
        cross2 = torch.cross(vec3, vec4)
        norm = (cross1.pow(2).sum(1) * cross2.pow(2).sum(1)).sqrt()
        cos_phi = 1.0 * ((cross1 * cross2).sum(1) / norm).view(-1, 1)
        pair1 = self.nonlinear(torch.cat([r[dihedrals[:, 1]], r[dihedrals[:, 0]]], dim=1))
        pair2 = self.nonlinear(torch.cat([r[dihedrals[:, 2]], r[dihedrals[:, 3]]], dim=1))
        dihedral_input = pair1 + pair2
        E = 0.0
        if 'multiharmonic' in self.terms:
            multiharmonic_constants = self.dihedralnet_multiharmonic(dihedral_input)
            for m in range(5):
                A = multiharmonic_constants[:, m].view(-1, 1)
                E = E + A * (cos_phi.pow(m))
            self.learned_params['multiharmonic']['dihedralnet'] = multiharmonic_constants.tolist()
        if 'OPLS' in self.terms:
            OPLS_constants = self.dihedralnet_OPLS(dihedral_input)
            phi = torch.acos(cos_phi / 1.000001)
            for m in range(4):
                V = OPLS_constants[:, m].view(-1, 1)
                E = E + (V / 2) * (1 + ((-1) ** m) * torch.cos((m + 1) * phi))
            self.learned_params['OPLS']['dihedralnet'] = OPLS_constants.tolist()

        # split the results into per-molecule energies through batch["num_dihedrals"]

        E = torch.stack([e.sum(0) for e in torch.split(E, num_dihedrals)])
        return (E)


class ImproperNet(torch.nn.Module):
    def __init__(self, Fr, Lh, terms=['harmonic'], trainable=False):
        super(ImproperNet, self).__init__()
        self.Fr = Fr
        self.Lh = Lh
        self.terms = terms
        self.true_params = None
        self.learned_params = {}
        self.nonlinear = ParameterPredictor(2 * self.Fr, Lh, Lh[-1], trainable=trainable)
        if 'harmonic' in self.terms:
            self.k_harmonic = ParameterPredictor(Lh[-1], Lh, 1, trainable=trainable)
            self.learned_params['harmonic'] = {'k': None}

    def forward(self, r, batch, xyz):

        impropers = batch["impropers"]
        num_impropers = batch["num_impropers"].tolist()

        if num_impropers == [0]*len(num_impropers):
            return torch.tensor([0.0 for _ in range(len(num_impropers))])

        N = xyz.shape[0]
        D = xyz.expand(N, N, 3) - xyz.expand(N, N, 3).transpose(0, 1)
        vec1 = D[impropers[:, 1], impropers[:, 0]]
        vec2 = D[impropers[:, 1], impropers[:, 2]]
        vec3 = D[impropers[:, 2], impropers[:, 1]]
        vec4 = D[impropers[:, 2], impropers[:, 3]]
        cross1 = torch.cross(vec1, vec2)
        cross2 = torch.cross(vec3, vec4)
        norm = (cross1.pow(2).sum(1) * cross2.pow(2).sum(1)).sqrt()
        cos_phi = 1.0 * ((cross1 * cross2).sum(1) / norm).view(-1, 1)
        phi = torch.acos(cos_phi / 1.000001)
        pair1 = self.nonlinear(torch.cat([r[impropers[:, 0]], r[impropers[:, 1]]], dim=1))
        pair2 = self.nonlinear(torch.cat([r[impropers[:, 0]], r[impropers[:, 2]]], dim=1))
        pair3 = self.nonlinear(torch.cat([r[impropers[:, 0]], r[impropers[:, 3]]], dim=1))
        improper_input = pair1 + pair2 + pair3
        E = 0.0

        if 'harmonic' in self.terms:
            k_harmonic = self.k_harmonic(improper_input).pow(2)
            E = E + (k_harmonic / 2) * (phi.pow(2))
            self.learned_params['harmonic']['k'] = k_harmonic.tolist()

        # split the results into per-molecule energies through batch["num_impropers"]

        E = torch.stack([e.sum(0) for e in torch.split(E, num_impropers)])
        return E


# class PairNet(torch.nn.Module):
#     def __init__(self, Fr, Lh, terms=['coulomb', 'LJ']):
#         super(PairNet, self).__init__()
#         self.Fr = Fr
#         self.Lh = Lh
#         self.terms = terms
#         self.true_params = None
#         self.learned_params = {}
#         if 'LJ' in self.terms:
#             self.sigma = ParameterPredictor(Fr, Lh, 1)
#             self.epsilon = ParameterPredictor(Fr, Lh, 1)
#             self.learned_params['LJ'] = {'sigma': None, 'epsilon': None}
#         if 'coulomb' in self.terms:
#             self.charge = ParameterPredictor(Fr, Lh, 1)
#             self.learned_params['coulomb'] = {'charge': None}
#         if 'induced_dipole' in self.terms:
#             self.alpha = ParameterPredictor(Fr, Lh, 1)
#             self.learned_params['induced_dipole'] = {'alpha': None}
#         self.coulomb_constant = torch.tensor([332.0636])
#         self.use_dft_charges = False
#         self.charges = None  ######################################################### rename to learned_charges?

#     def forward(self, batch):

#         r = batch["r"]
#         xyz = batch["xyz"]
#         pairs = batch["pairs"]
#         # will need to fix this:
#         num_pairs = len(pairs)

#         # pairs = topology['pair'] # ['indices']
#         # num_pairs = topology['pair']['num']

#         # self.true_params = topology['true_params']['pair']
#         try:
#             # _1_4_pairs = topology['dihedral']['indices'][:, [0, 3]]
#             _1_4_pairs = topology['dihedral'][:, [0, 3]]
#             _1_4_pair_indices = (pairs.unsqueeze(1) - _1_4_pairs.unsqueeze(0)).abs().sum(2)
#             _1_4_pair_indices = (_1_4_pair_indices == 0).nonzero()[:, 0].unique()
#         except:
#             _1_4_pair_indices = []
#         N = xyz.shape[0]
#         displacements = xyz.unsqueeze(1).expand(N, N, 3)
#         displacements = -(displacements - displacements.transpose(0, 1))
#         # displacements[i,j] is the vector from atom i to  atom j (xyz_j - xyz_i)
#         D2 = displacements.pow(2).sum(2)
#         D2 = D2[pairs[:, 0], pairs[:, 1]].view(-1, 1)
#         inv_D = D2.pow(-0.5)
#         E = 0.0
#         if 'coulomb' in self.terms:
#             # if self.use_dft_charges:
#             #     self.charges = self.true_params[:, [0]]
#             #     charge_product = self.charges[pairs].prod(1)
#             # else:
#             self.charges = self.charge(r)
#             charge_product = self.charges[pairs].prod(1)
#             E = E + charge_product * inv_D * self.coulomb_constant.to(r.device)
#             self.learned_params['coulomb']['charge'] = self.charges.tolist()
#             if 'induced_dipole' in self.terms:
#                 # Method: Induced Point Dipole Model with Thole Damping
#                 # Reference: Antila H.S., Salonen E. (2013) Polarizable Force Fields.
#                 # In: Monticelli L., Salonen E. (eds) Biomolecular Simulations.
#                 # Methods in Molecular Biology (Methods and Protocols), vol 924.
#                 # (pp 215-241). Humana Press, Totowa, NJ.
#                 # if use_true_params:
#                 #     raise Exception("Figure out formatting for true alpha values.")
#                 # else:
                    
#                 alpha = self.alpha(r).pow(2)
#                 q = self.charges.unsqueeze(2).expand(N, N, 3)
#                 # Add ones along the diagonal to avoid division by zero
#                 Displacements = displacements + (1 / np.sqrt(3)) * torch.eye(N).unsqueeze(2).expand(N, N, 3).to(
#                     r.device)
#                 E_field = q * Displacements / (Displacements.pow(2).sum(2, keepdim=True).pow(1.5))
#                 # E_field[i,j] is the electric field at atom j due to atom i
#                 # After the following manipulations, E_field[i] will be the total electric field
#                 # at atom i due to the permanent partial charges at all other atoms (j != i) with
#                 # which it has a pair interaction
#                 pair_mask = torch.zeros(N, N).to(r.device)
#                 pair_mask[pairs[:, 0], pairs[:, 1]] += 1
#                 pair_mask = pair_mask + pair_mask.t()
#                 pair_mask = pair_mask.unsqueeze(2).expand(N, N, 3)
#                 E_field = pair_mask * E_field
#                 E_field = E_field.sum(0)
#                 # Now that I have the electric field at each atom due to the permanent charges of all
#                 # other atoms with which it has a pair interaction, I need to consider dipole-dipole
#                 # interactions. I will construct the shape function tensor t, the interaction tensor
#                 # T, and the polarizability tensor Alpha, from which the dipole tensor mu can be
#                 # calculated by matrix inversion as described in the reference above.
#                 R = Displacements.pow(2).sum(2).pow(0.5)
#                 # Finally, I will re-zero the diagonal elements of the distance tensor
#                 R = R * (1 - torch.eye(N, N)).to(r.device)
#                 # Thole Model for Damped Interactions
#                 # To clarify my approach, I will define a few new terms. See Equation (21) at the
#                 # reference above. I will treat these two terms separately, and will further
#                 # break down each term as defined by: t = tau1*delta1 + mixed_alpha*tau2*delta2,
#                 # where tau1 and tau2 are the tensor elements which depend only on atom indices,
#                 # delta1 and delta2 are the tensors elements which depend on both cartesian indices
#                 # and atom indices, and mixed_alpha is a pre-factor for the second term which
#                 # depends only on the atom indices.
#                 a = 0.39
#                 mixed_alpha = (alpha.expand(N, N) * alpha.expand(N, N).t()).pow(1. / 6)
#                 u = R / mixed_alpha
#                 tau1 = u.pow(3) * (1 - ((a ** 2) * u.pow(2) / 2. + a * u + 1) * torch.exp(-a * u))
#                 tau1 = tau1 + torch.eye(N, N).to(r.device)
#                 tau1 = 1. / tau1
#                 tau1 = tau1 - torch.eye(N, N).to(r.device)
#                 tau2 = R.pow(5) * (
#                             1 - ((a ** 3) * R.pow(3) / 6. + (a ** 2) * R.pow(2) / 2. + a * R + 1) * torch.exp(-a * u))
#                 tau2 = tau2 + torch.eye(N, N).to(r.device)
#                 tau2 = 1. / tau2
#                 tau2 = tau2 - torch.eye(N, N).to(r.device)
#                 tau2 = -3. * tau2
#                 delta1 = torch.eye(3).view(1, 1, 3, 3).expand(N, N, 3, 3).to(r.device)
#                 delta1 = delta1 * (1 - torch.eye(N)).view(N, N, 1, 1).to(r.device)
#                 indices = torch.ones(3, 3).nonzero().view(3, 3, 2)
#                 delta2 = displacements[:, :, indices[:, :, 0]] * displacements[:, :, indices[:, :, 1]]
#                 tau1 = tau1.view(N, N, 1, 1)
#                 tau2 = tau2.view(N, N, 1, 1)
#                 mixed_alpha = mixed_alpha.view(N, N, 1, 1)
#                 t = tau1 * delta1 + mixed_alpha * tau2 * delta2
#                 T = t / ((alpha.expand(N, N) * alpha.expand(N, N).t()).pow(0.5).view(N, N, 1, 1))
#                 T = T.transpose(1, 2).reshape(3 * N, 3 * N)
#                 Alpha = torch.diag(alpha.expand(N, 3).reshape(-1))
#                 A = Alpha.inverse() + T
#                 B = A.inverse()
#                 mu = torch.matmul(B, E_field.view(-1, 1)).view(N, 3)
#                 # With the induced dipoles known, I can now calculate the interaction energies
#                 E_dipole = -0.5 * (mu * E_field).sum(1) * self.coulomb_constant.to(r.device)
#                 E_dipole = E_dipole.view(-1, 1)
#                 E_dipole = torch.stack([e.sum(0) for e in torch.split(E_dipole, N)])
#                 self.learned_params['induced_dipole']['alpha'] = alpha.tolist()
#             else:
#                 E_dipole = 0.0
#         else:
#             E_dipole = 0.0
#         if 'LJ' in self.terms:
#             # if use_true_params:
#             #     sigma = self.true_params[:, [2]]
#             #     epsilon = self.true_params[:, [1]]
#             # else:
#                 sigma = ((self.true_params[:, [2]] ** 0.5) + 0.1 * self.sigma(r)).pow(2)
#                 epsilon = ((self.true_params[:, [1]] ** 0.5) + 0.01 * self.epsilon(r)).pow(2)
#             sigma_mixed = sigma[pairs].prod(1).pow(0.5)
#             epsilon_mixed = epsilon[pairs].prod(1).pow(0.5)
#             D_inv_scal = sigma_mixed * inv_D
#             E = E + (4 * epsilon_mixed * (D_inv_scal.pow(12) - D_inv_scal.pow(6)))
#             self.learned_params['LJ']['sigma'] = sigma.tolist()
#             self.learned_params['LJ']['epsilon'] = epsilon.tolist()
#         f = torch.ones_like(E).to(torch.float)
#         f[_1_4_pair_indices] *= 0.5
#         E = f * E
#         E = torch.stack([e.sum(0) for e in torch.split(E, num_pairs)])
#         E = E + E_dipole
#         return (E)


TopologyNet = {
    'bond': BondNet,
    'angle': AngleNet,
    'dihedral': DihedralNet,
    'improper': ImproperNet,
    # 'pair': PairNet
}

class AuTopologyReadOut(nn.Module):

    """
    Class for reading out results from a convolution using AuTopology.
    Attributes:
        terms (dict): dictionary of the types of AuTopology potentials used
            for each kind of topology (e.g. Morse for harmonic, LJ for pairs,
            etc.)
        auto_modules (torch.nn.ModuleDict): module dictionary for all the topology
            nets associated with each energy state. E.g. of the form {"energy_0":
            {"bond": BondNet0, "angle": AngletNet0}, "energy_1": {"bond": BondNet1,
            "angle": AngletNet1} }.
    """

    def __init__(self, multitaskdict):

        """
        Args:
            multitaskdict (dict): dictionary of items used for setting up the networks.
        Returns:
            None
        """

        super(AuTopologyReadOut, self).__init__()

        trainable = multitaskdict["trainable_prior"]
        Fr = multitaskdict["Fr"]
        Lh = multitaskdict["Lh"]
        bond_terms = multitaskdict.get("bond_terms", ["morse"])
        angle_terms =  multitaskdict.get("angle_terms", ['harmonic'])  # harmonic and/or cubic and/or quartic
        dihedral_terms = multitaskdict.get("dihedral_terms", ['OPLS'])  # OPLS and/or multiharmonic
        improper_terms = multitaskdict.get("improper_terms", ['harmonic'])  # harmonic
        pair_terms = multitaskdict.get("pair_terms", ['coulomb', 'LJ'])  # coulomb and/or LJ and/or induced_dipole
        autopology_keys = multitaskdict["sorted_result_keys"]


        self.terms = {
            'bond': bond_terms,
            'angle': angle_terms,
            'dihedral': dihedral_terms,
            'improper': improper_terms
            # 'pair': pair_terms
        }


        topologynet = {key: {} for key in autopology_keys}
        for key in autopology_keys:
            for top in self.terms.keys():
                topologynet[key][top] = TopologyNet[top](Fr, Lh, self.terms[top], trainable=trainable)

        # module dictionary of the form {"energy_0": {"bond": BondNet0, "angle": AngletNet0},
        # "energy_1": {"bond": BondNet1, "angle": AngletNet1} }
        self.auto_modules = ModuleDict({key: ModuleDict({top: topologynet[key][top] for top in
            self.terms.keys()}) for key in autopology_keys})


    def forward(self, r, batch, xyz, take_grad=True):

        output = dict()

        # loop through output keys (e.g. energy_0 and energy_1)
        for output_key, top_set in self.auto_modules.items():
            E = {key: 0.0 for key in list(self.terms.keys()) + ['total']}
            learned_params = {}
            # loop through associated topology nets (e.g. BondNet0 and AngletNet0 or
            # BondNet1 and AngletNet1)
            for top, top_net in top_set.items():
                E[top] = top_net(r, batch, xyz)
                learned_params[top] = top_net.learned_params
                E['total'] += E[top]
            output[output_key] = E["total"] 
            if take_grad:
                grad = compute_grad(inputs=xyz, output=E["total"] )
                output[output_key + "_grad"] = grad

        return output



class SchNetEdgeUpdate(EdgeUpdateModule):
    """
    Arxiv.1806.03146

    Attributes:
        mlp (TYPE): Update function
    """

    def __init__(self, n_atom_basis):
        super(SchNetEdgeUpdate, self).__init__()

        self.mlp = Sequential(
            Linear(2 * n_atom_basis, n_atom_basis),
            ReLU(),  # softplus in the original paper
            Linear(n_atom_basis, n_atom_basis),
            ReLU(),  # softplus in the original paper
            Linear(n_atom_basis, 1)
        )

    def aggregate(self, message, neighborlist):
        aggregated_edge_feature = torch.cat((message[neighborlist[:, 0]], message[neighborlist[:, 1]]), 1)
        return aggregated_edge_feature

    def update(self, e):
        return self.mlp(e)


class SchNetConv(MessagePassingModule):
    """The convolution layer with filter. To be merged with GraphConv class.

    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(self,
                 n_atom_basis,
                 n_filters,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict({
            'message_edge_filter': Sequential(
                GaussianSmearing(
                    start=0.0,
                    stop=cutoff,
                    n_gaussians=n_gaussians,
                    trainable=trainable_gauss
                ),
                Dense(in_features=n_gaussians, out_features=n_gaussians),
                shifted_softplus(),
                Dense(in_features=n_gaussians, out_features=n_filters)
            ),
            'message_node_filter': Dense(in_features=n_atom_basis, out_features=n_filters),
            'update_function': Sequential(
                Dense(in_features=n_filters, out_features=n_atom_basis),
                shifted_softplus(),
                Dense(in_features=n_atom_basis, out_features=n_atom_basis)
            )
        })

    def message(self, r, e, a):
        """The message function for SchNet convoltuions


        Args:
            r (TYPE): node inputs
            e (TYPE): edge inputs
            a (TYPE): neighbor list

        Returns:
            TYPE: message should a pair of message and
        """
        # update edge feature
        e = self.moduledict['message_edge_filter'](e)
        # convection: update
        r = self.moduledict['message_node_filter'](r)
        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict['update_function'](r)


class GraphAttention(MessagePassingModule):
    """Weighted graph pooling layer based on self attention

    Attributes:
        activation (TYPE): Description
        weight (TYPE): Description
    """

    def __init__(self, n_atom_basis):
        super(GraphAttention, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(1, 2 * n_atom_basis))
        self.activation = LeakyReLU()

    def message(self, r, e, a):
        """weight_ij is the importance factor of node j to i
           weight_ji is the importance factor of node i to j

        Args:
            r (TYPE): Description
            e (TYPE): Description
            a (TYPE): Description

        Returns:
            TYPE: Description
        """
        # i -> j
        weight_ij = torch.exp(self.activation(torch.cat((r[a[:, 0]], r[a[:, 1]]), dim=1) * \
                                              self.weight).sum(-1))
        # j -> i
        weight_ji = torch.exp(self.activation(torch.cat((r[a[:, 1]], r[a[:, 0]]), dim=1) * \
                                              self.weight).sum(-1))

        weight_ii = torch.exp(self.activation(torch.cat((r, r), dim=1) * \
                                              self.weight).sum(-1))

        normalization = scatter_add(weight_ij, a[:, 0], dim_size=r.shape[0]) \
                        + scatter_add(weight_ji, a[:, 1], dim_size=r.shape[0]) + weight_ii

        a_ij = weight_ij / normalization[a[:, 0]]  # the importance of node j’s features to node i
        a_ji = weight_ji / normalization[a[:, 1]]  # the importance of node i’s features to node j
        a_ii = weight_ii / normalization  # self-attention

        message = r[a[:, 0]] * a_ij[:, None], \
                  r[a[:, 1]] * a_ij[:, None], \
                  r * a_ii[:, None]

        return message

    def forward(self, r, e, a):
        # Base case
        graph_size = r.shape[0]

        rij, rji, r = self.message(r, e, a)

        # i -> j propagate
        r += self.aggregate(rij, a[:, 1], graph_size)
        # j -> i propagate
        r += self.aggregate(rji, a[:, 0], graph_size)

        r = self.update(r)

        return r


class NodeMultiTaskReadOut(nn.Module):
    """Stack Multi Task outputs

        example multitaskdict:

        multitaskdict = {
            'myenergy_0': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'myenergy_1': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'muliken_charges': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ]
        }

        example post_readout:

        def post_readout(predict_dict, readoutdict):
            sorted_keys = sorted(list(readoutdict.keys()))
            sorted_ens = torch.sort(torch.stack([predict_dict[key] for key in sorted_keys]))[0]
            sorted_dic = {key: val for key, val in zip(sorted_keys, sorted_ens) }
            return sorted_dic
    """

    def __init__(self, multitaskdict, post_readout=None):
        """Summary

        Args:
            multitaskdict (dict): dictionary that contains model information
        """
        super(NodeMultiTaskReadOut, self).__init__()
        # construct moduledict
        self.readout = construct_module_dict(multitaskdict)
        self.post_readout = post_readout
        self.multitaskdict = multitaskdict

    def forward(self, r):
        predict_dict = dict()
        for key in self.readout:
            predict_dict[key] = self.readout[key](r)
        if self.post_readout is not None:
            predict_dict = self.post_readout(predict_dict, self.multitaskdict)

        return predict_dict


class GraphDis(GeometricOperations):
    """Compute distance matrix on the fly

    Attributes:
        box_size (numpy.array): Length of the box, dim = (3, )
        cutoff (float): cutoff for convolution
        F (int): Fr + Fe
        Fe (int): edge feature length
        Fr (int): node feature length
    """

    def __init__(
            self,
            cutoff,
            box_size=None
    ):
        super(GraphDis, self).__init__()

        self.cutoff = cutoff

        if box_size is not None:
            self.box_size = torch.Tensor(box_size)
        else:
            self.box_size = None

    def get_bond_vector_matrix(self, frame):
        """A function to compute the distance matrix

        Args:
            frame (torch.FloatTensor): coordinates of (B, N, 3)

        Returns:
            torch.FloatTensor: distance matrix of dim (B, N, N, 1)
        """
        device = frame.device

        n_atoms = frame.shape[0]
        frame = frame.view(-1, n_atoms, 1, 3)
        dis_mat = frame.expand(-1, n_atoms, n_atoms, 3) \
                  - frame.expand(-1, n_atoms, n_atoms, 3).transpose(1, 2)

        if self.box_size is not None:
            box_size = self.box_size.to(device)

            # build minimum image convention
            box_size = self.box_size
            mask_pos = dis_mat.ge(0.5 * box_size).float()
            mask_neg = dis_mat.lt(-0.5 * box_size).float()

            # modify distance
            dis_add = mask_neg * box_size
            dis_sub = mask_pos * box_size
            dis_mat = dis_mat + dis_add - dis_sub

        # create cutoff mask

        # compute squared distance of dim (B, N, N)
        dis_sq = dis_mat.pow(2).sum(3)

        # mask is a byte tensor of dim (B, N, N)
        mask = (dis_sq <= self.cutoff ** 2) & (dis_sq != 0)

        A = mask.unsqueeze(3).float()

        # 1) PBC 2) # gradient of zero distance
        dis_sq = dis_sq.unsqueeze(3)

        # to make sure the distance is not zero
        # otherwise there will be inf gradient
        dis_sq = (dis_sq * A) + EPSILON
        dis_mat = dis_sq.sqrt()

        return dis_mat, A.squeeze(3)

    def forward(self, xyz):
        frame = xyz
        e, A = self.get_bond_vector_matrix(frame=frame)

        n_atoms = frame.shape[0]

        #  use only upper triangular to generative undirected adjacency matrix
        A = A * torch.ones(n_atoms, n_atoms).triu()[None, :, :].to(A.device)
        e = e * A.unsqueeze(-1)

        # compute neighbor list
        a = A.nonzero()

        # reshape distance list
        e = e[a[:, 0], a[:, 1], a[:, 2], :].reshape(-1, 1)

        # reindex neighbor list
        a = (a[:, 0] * n_atoms)[:, None] + a[:, 1:3]

        return e, a


class BondEnergyModule(nn.Module):

    def __init__(self, batch=True):
        super().__init__()

    def forward(self, xyz, bond_adj, bond_len, bond_par):
        e = (
                    xyz[bond_adj[:, 0]] - xyz[bond_adj[:, 1]]
            ).pow(2).sum(1).sqrt()[:, None]

        ebond = bond_par * (e - bond_len) ** 2
        energy = 0.5 * scatter_add(src=ebond, index=bond_adj[:, 0], dim=0, dim_size=xyz.shape[0])
        energy += 0.5 * scatter_add(src=ebond, index=bond_adj[:, 1], dim=0, dim_size=xyz.shape[0])

        return energy


# Test

class TestModules(unittest.TestCase):

    def testBaseEdgeUpdate(self):
        # initialize basic graphs
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e = torch.rand(5, 10)
        r_in = torch.rand(6, 10)
        model = MessagePassingModule()
        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape, "The node feature dimensions should be same for the base case")

    def testBaseMessagePassing(self):
        # initialize basic graphs
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e_in = torch.rand(5, 10)
        r = torch.rand(6, 10)
        model = EdgeUpdateModule()
        e_out = model(r, e_in, a)
        self.assertEqual(e_in.shape, e_out.shape, "The edge feature dimensions should be same for the base case")

    def testSchNetMPNN(self):
        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        # SchNet params
        n_atom_basis = 10
        n_filters = 10
        n_gaussians = 10
        num_nodes = 6
        cutoff = 0.5

        e = torch.rand(5, n_atom_basis)
        r_in = torch.rand(num_nodes, n_atom_basis)

        model = SchNetConv(
            n_atom_basis,
            n_filters,
            n_gaussians,
            cutoff=2.0,
            trainable_gauss=False,
        )

        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape,
                         "The node feature dimensions should be same for the SchNet Convolution case")

    def testSchNetEdgeUpdate(self):
        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        # SchNet params
        n_atom_basis = 10
        num_nodes = 6

        e_in = torch.rand(5, 1)
        r = torch.rand(num_nodes, n_atom_basis)

        model = SchNetEdgeUpdate(n_atom_basis=n_atom_basis)
        e_out = model(r, e_in, a)

        self.assertEqual(e_in.shape, e_out.shape,
                         "The edge feature dimensions should be same for the SchNet Edge Update case")

    def testGAT(self):
        n_atom_basis = 10

        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e = torch.rand(5, n_atom_basis)
        r_in = torch.rand(6, n_atom_basis)

        attention = GraphAttention(n_atom_basis=n_atom_basis)

        r_out = attention(r_in, e, a)

        self.assertEqual(r_out.shape, r_in.shape)

    def testmultitask(self):
        n_atom = 10
        r = torch.rand(n_atom, 5)

        multitaskdict = {
            "myenergy0":
                [
                    {'name': 'Dense', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'shifted_softplus', 'param': {}},
                    {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                ],

            "myenergy1":
                [
                    {'name': 'linear', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                ],
            "Muliken charges":
                [
                    {'name': 'linear', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'linear', 'param': {'in_features': 20, 'out_features': 1}}
                ]
        }

        model = NodeMultiTaskReadOut(multitaskdict)
        output = model(r)


if __name__ == '__main__':
    unittest.main()



