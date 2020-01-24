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
TOPS = ['bond', 'angle', 'dihedral', 'improper', 'pair']


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

class PairNet(torch.nn.Module):

    """
    Only Lennard-Jones for now
    """

    def __init__(self, Fr, Lh, terms=['LJ'], trainable=False):
        super(PairNet, self).__init__()
        self.Fr = Fr
        self.Lh = Lh
        self.terms = terms
        self.learned_params = dict()

        if 'LJ' in self.terms:
            self.sigma = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.epsilon = ParameterPredictor(Fr, Lh, 1, trainable=trainable)
            self.learned_params['LJ'] = {'sigma': None, 'epsilon': None}

    
    def forward(self, r, batch, xyz):

        pairs = batch["pairs"]
        num_pairs = batch["num_pairs"].tolist()

        N = xyz.shape[0]
        displacements = xyz.unsqueeze(1).expand(N,N,3)
        displacements = -(displacements-displacements.transpose(0,1))
        # displacements[i,j] is the vector from atom i to  atom j (xyz_j - xyz_i)
        D2 = displacements.pow(2).sum(2)
        D2 = D2[pairs[:,0], pairs[:,1]].view(-1,1)
        inv_D = D2.pow(-0.5)

        E = 0.0*inv_D

        if 'LJ' in self.terms:


            sigma = 4.0+10*self.sigma(r).pow(2)
            epsilon = 0.1*self.epsilon(r).pow(2)

            sigma_mixed = sigma[pairs].prod(1).pow(0.5)
            epsilon_mixed = epsilon[pairs].prod(1).pow(0.5)

            D_inv_scal = sigma_mixed*inv_D
            E = E + (4*epsilon_mixed*(D_inv_scal.pow(12)-D_inv_scal.pow(6)))

            self.learned_params["LJ"]["sigma"] = sigma
            self.learned_params["LJ"]["epsilon"] = epsilon

        E = torch.stack([e.sum(0) for e in torch.split(E, num_pairs)])

        return E


TopologyNet = {
    'bond': BondNet,
    'angle': AngleNet,
    'dihedral': DihedralNet,
    'improper': ImproperNet,
    'pair': PairNet
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
        # bond_terms = multitaskdict.get("bond_terms", ["morse"])
        # angle_terms =  multitaskdict.get("angle_terms", ['harmonic'])  # harmonic and/or cubic and/or quartic
        # dihedral_terms = multitaskdict.get("dihedral_terms", ['OPLS'])  # OPLS and/or multiharmonic
        # improper_terms = multitaskdict.get("improper_terms", ['harmonic'])  # harmonic
        # pair_terms = multitaskdict.get("pair_terms", ['LJ'])  # coulomb and/or LJ and/or induced_dipole
        autopology_keys = multitaskdict["output_keys"]

        default_terms_dict = {
            "bond_terms": ["morse"],
            "angle_terms": ["harmonic"],
            "dihedral_terms": ["OPLS"],
            "improper_terms": ["harmonic"],
            "pair_terms": ["LJ", "coulombs"]
        }

        # self.terms = {
        #     'bond': bond_terms,
        #     'angle': angle_terms,
        #     'dihedral': dihedral_terms,
        #     'improper': improper_terms,
        #     'pair': pair_terms
        # }
        self.terms = {}

        # remove terms that is not included 
        for top in ['bond', 'angle', 'dihedral', 'improper', 'pair']:
            if top + '_terms' in multitaskdict.keys():
                self.terms[top] = multitaskdict.get(top + '_terms', default_terms_dict[top + '_terms'])


        topologynet = {key: {} for key in autopology_keys}
        for key in autopology_keys:
            for top in self.terms.keys():
                if top + '_terms' in multitaskdict:
                    topologynet[key][top] = TopologyNet[top](Fr, Lh, self.terms[top], trainable=trainable)


        # module dictionary of the form {"energy_0": {"bond": BondNet0, "angle": AngletNet0},
        # "energy_1": {"bond": BondNet1, "angle": AngletNet1} }
        self.auto_modules = ModuleDict({key: ModuleDict({top: topologynet[key][top] for top in
            self.terms.keys()}) for key in autopology_keys})

        # energy offset for each state
        self.offset = ModuleDict({key: ParameterPredictor(Fr, Lh, 1)
            for key in autopology_keys})


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

            N = batch["num_atoms"].cpu().numpy().tolist()
            offset = torch.split(self.offset[output_key](r), N)
            offset = (torch.stack([torch.sum(item) for item in offset])).reshape(-1, 1)

            output[output_key] = E["total"] + offset

            if take_grad:
                grad = compute_grad(inputs=xyz, output=E["total"])
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

    """The convolution layer with filter.
    
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

    def message(self, r, e, a, aggr_wgt=None):
        """The message function for SchNet convoltuions 
        Args:
            r (TYPE): node inputs
            e (TYPE): edge inputs
            a (TYPE): neighbor list
            aggr_wgt (None, optional): Description

        Returns:
            TYPE: message should a pair of message and
        """
        # update edge feature
        e = self.moduledict['message_edge_filter'](e)
        # convection: update
        r = self.moduledict['message_node_filter'](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict['update_function'](r)



class AuTopologyConv(MessagePassingModule):

    """
    Base class for AuTopology convolutions.
    Attributes:
        update_function (nn.Module): network to update features after convolution

    """

    def __init__(self, update_layers):
        super(AuTopologyConv, self).__init__()

        """
        Args:
            update_layers (dict): dictionary of layers to apply after the convolution
        Returns:
            None
        Example:
                update_layers =  [{'name': 'linear', 'param' : {'in_features': 256,
                                                                        'out_features': 256}},
                                          {'name': 'tanh', 'param': {}},
                                          {'name': 'linear', 'param' : {'in_features': 256,
                                                                  'out_features': 256}},
                                          {'name': 'tanh', 'param': {}}
        """

        self.update_function = construct_sequential(update_layers)

    def update(self, r):
        return self.update_function(r)

    def message(self, r, e, a):
        raise NotImplementedError


class DoubleNodeConv(AuTopologyConv):

    """

    AuTopology convolution that uses a features of bonded nodes and center nodes.

    """

    def __init__(self, update_layers):

        """
        Args:
            update_layers (dict): dictionary of layers to apply after the convolution
        Returns:
            None
        """

        super(DoubleNodeConv, self).__init__(update_layers)

    def message(self, r, e, a):
        """ 
        Get the message:
        Args:
            r (tensor): feature matrix
            e (None): edge features (not used)
            a (tensor): bonded neighbor list
        """
        # the message is simply the features of all the atoms bonded to the current atom
        return r[a[:, 0]], r[a[:, 1]]

    def forward(self, r, e, a):

        rij, rji = self.message(r, e, a)

        graph_size = r.shape[0]
        # sum over all bonded node features
        bonded_node_sum = self.aggregate(rij, a[:, 1], graph_size)
        bonded_node_sum += self.aggregate(rji, a[:, 0], graph_size)

        # sum the features of this node once for each of its bonds 
        this_node_sum = self.aggregate(r[a[:, 0]], a[:, 0], graph_size)
        this_node_sum += self.aggregate(r[a[:, 1]], a[:, 1], graph_size)

        # the new feature vector is a concatenation
        new_r = torch.cat([bonded_node_sum, this_node_sum], dim=-1)
        new_r = self.update(new_r)

        return new_r

class SingleNodeConv(AuTopologyConv):

    """

    AuTopology convolution that adds the features of bonded nodes without concatenating them with
    the features of the center node.

    """

    def __init__(self, update_layers):

        """
        Args:
            update_layers (dict): dictionary of layers to apply after the convolution
        Returns:
            None
        """

        super(SingleNodeConv, self).__init__(update_layers)

    def message(self, r, e, a):
        """ 
        Get the message:
        Args:
            r (tensor): feature matrix
            e (None): edge features (not used)
            a (tensor): bonded neighbor list
        """
        # the message is simply the features of all the atoms bonded to the current atom

        return r[a[:, 0]], r[a[:, 1]]



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
                         "The node feature dimensions should be same.")


    def testDoubleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers =  [{'name': 'linear', 'param' : {'in_features': 2*num_features,
                                                        'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}},
                          {'name': 'linear', 'param' : {'in_features': num_features,
                                                  'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}}]


        r_in = torch.rand(num_nodes, num_features)
        model = DoubleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

        self.assertEqual(r_in.shape, r_out.shape,
                         "The node feature dimensions should be same.")

    def testSingleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers =  [{'name': 'linear', 'param' : {'in_features': num_features,
                                                        'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}},
                          {'name': 'linear', 'param' : {'in_features': num_features,
                                                  'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}}]


        r_in = torch.rand(num_nodes, num_features)
        model = SingleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

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



