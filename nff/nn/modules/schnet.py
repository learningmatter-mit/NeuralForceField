
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict
from torch.nn import Softmax
from torch.nn.functional import softmax

import numpy as np
import unittest

from nff.nn.layers import Dense, GaussianSmearing, StochasticIncrease
from nff.utils.scatter import scatter_add, compute_grad
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import (
    MessagePassingModule,
    EdgeUpdateModule,
)
from nff.nn.utils import (construct_sequential, construct_module_dict,
                          chemprop_msg_update, chemprop_msg_to_node,
                          remove_bias)
from nff.nn.layers import Diagonalize
from nff.utils.tools import layer_types


EPSILON = 1e-15
DEFAULT_BONDPRIOR_PARAM = {"k": 20.0}


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
            Linear(n_atom_basis, 1),
        )

    def aggregate(self, message, neighborlist):
        aggregated_edge_feature = torch.cat(
            (message[neighborlist[:, 0]], message[neighborlist[:, 1]]), 1
        )
        return aggregated_edge_feature

    def update(self, e):
        return self.mlp(e)


class SchNetEdgeFilter(nn.Module):
    def __init__(self,
                 cutoff,
                 n_gaussians,
                 trainable_gauss,
                 n_filters,
                 dropout_rate,
                 activation='shifted_softplus'):

        super(SchNetEdgeFilter, self).__init__()

        self.filter = Sequential(
            GaussianSmearing(
                start=0.0,
                stop=cutoff,
                n_gaussians=n_gaussians,
                trainable=trainable_gauss,
            ),
            Dense(
                in_features=n_gaussians,
                out_features=n_gaussians,
                dropout_rate=dropout_rate,
            ),
            layer_types[activation](),
            Dense(
                in_features=n_gaussians,
                out_features=n_filters,
                dropout_rate=dropout_rate,
            ))

    def forward(self, e):
        return self.filter(e)


class SchNetConv(MessagePassingModule):

    """The convolution layer with filter.

    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff,
        trainable_gauss,
        dropout_rate,
    ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {
                "message_edge_filter": Sequential(
                    GaussianSmearing(
                        start=0.0,
                        stop=cutoff,
                        n_gaussians=n_gaussians,
                        trainable=trainable_gauss,
                    ),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_gaussians,
                        dropout_rate=dropout_rate,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_gaussians,
                        out_features=n_filters,
                        dropout_rate=dropout_rate,
                    ),
                ),
                "message_node_filter": Dense(
                    in_features=n_atom_basis,
                    out_features=n_filters,
                    dropout_rate=dropout_rate,
                ),
                "update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

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
        e = self.moduledict["message_edge_filter"](e)
        # convection: update
        r = self.moduledict["message_node_filter"](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * \
            e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)


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
        weight_ij = torch.exp(
            self.activation(
                torch.cat((r[a[:, 0]], r[a[:, 1]]), dim=1) * self.weight
            ).sum(-1)
        )
        # j -> i
        weight_ji = torch.exp(
            self.activation(
                torch.cat((r[a[:, 1]], r[a[:, 0]]), dim=1) * self.weight
            ).sum(-1)
        )

        weight_ii = torch.exp(
            self.activation(torch.cat((r, r), dim=1) * self.weight).sum(-1)
        )

        normalization = (
            scatter_add(weight_ij, a[:, 0], dim_size=r.shape[0])
            + scatter_add(weight_ji, a[:, 1], dim_size=r.shape[0])
            + weight_ii
        )

        a_ij = (
            weight_ij / normalization[a[:, 0]]
        )  # the importance of node j’s features to node i
        a_ji = (
            weight_ji / normalization[a[:, 1]]
        )  # the importance of node i’s features to node j
        a_ii = weight_ii / normalization  # self-attention

        message = (
            r[a[:, 0]] * a_ij[:, None],
            r[a[:, 1]] * a_ij[:, None],
            r * a_ii[:, None],
        )

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


class BondPrior(torch.nn.Module):
    def __init__(self, modelparams=DEFAULT_BONDPRIOR_PARAM):
        torch.nn.Module.__init__(self)
        self.k = modelparams["k"]

    def forward(self, batch):

        result = {}

        num_bonds = batch["num_bonds"].tolist()

        xyz = batch["nxyz"][:, 1:4]
        xyz.requires_grad = True
        bond_list = batch["bonds"]

        r_0 = batch["bond_len"].squeeze()

        r = (xyz[bond_list[:, 0]] - xyz[bond_list[:, 1]]).pow(2).sum(-1).sqrt()

        e = self.k * (r - r_0).pow(2)

        E = torch.stack([e.sum(0).reshape(1)
                         for e in torch.split(e, num_bonds)])

        result["energy"] = E
        result["energy_grad"] = compute_grad(inputs=xyz, output=E)

        return result


class MixedSchNetConv(MessagePassingModule):

    """
    SchNet convolution applied to edge features from both
    distances and bond features. 
    """

    def __init__(
        self,
        n_atom_hidden,
        n_filters,
        dropout_rate,
        n_bond_hidden,
        activation='shifted_softplus'
    ):
        """
        Args:
            n_atom_hidden (int): hidden dimension of the atom
                features. Same as `n_atom_basis` in regular
                SchNet, but different than `n_atom_basis` in
                SchNetFeatures, where `n_atom_basis` is the initial
                dimension of the atom feature vector and
                `n_atom_hidden` is its dimension after going through
                another network.
            n_filters (int): dimension of the distance hidden vector
            dropout_rate (float): dropout rate
            n_bond_hidden (int): dimension of the bond hidden vector
            activation (str): nonlinear activation name
        Returns:
            None
        """
        super(MixedSchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {

                # convert the atom features to the dimension
                # of cat(hidden_distance, hidden_bond)
                "message_node_filter": Dense(
                    in_features=n_atom_hidden,
                    out_features=(n_filters + n_bond_hidden),
                    dropout_rate=dropout_rate,
                ),
                # after multiplying edge features with
                # node features, convert them back to size
                # `n_atom_hidden`
                "update_function": Sequential(
                    Dense(
                        in_features=(n_filters + n_bond_hidden),
                        out_features=n_atom_hidden,
                        dropout_rate=dropout_rate,
                    ),
                    layer_types[activation](),
                    Dense(
                        in_features=n_atom_hidden,
                        out_features=n_atom_hidden,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

    def message(self, r, e, a):
        """The message function for SchNet convoltuions 
        Args:
            r (torch.Tensor): node inputs
            e (torch.Tensor): edge inputs
            a (torch.Tensor): neighbor list

        Returns:
            message (torch.Tensor): message from adjacent
                atoms.
        """
        # convection: update
        r = self.moduledict["message_node_filter"](r)
        # assumes directed neighbor list; no need
        # to supplement with r[a[:, 1]]
        message = r[a[:, 0]] * e
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)

    def forward(self, r, e, a):

        graph_size = r.shape[0]
        rij = self.message(r, e, a)
        r = self.aggregate(rij, a[:, 1], graph_size)
        r = self.update(r)
        return r


class ConfAttention(nn.Module):
    """
    Module to apply an attention mechanism to different conformers
    of a species to learn their respective contribution to the total
    fingerprint.
    """

    def __init__(self,
                 mol_basis,
                 boltz_basis,
                 final_act,
                 equal_weights=False,
                 prob_func='softmax'):
        """
        Args:
            mol_basis (int): dimension of the molecular fingerprint
            boltz_basis (int): dimension into which we embed the boltzmann
                weight as a vector.
            final_act (str): name of the final nonlinear layer to apply to
                the fingerprint.
            equal_weights (bool): whether to not use attention and just use
                equal weights for each conformer.
            prob_func (str): fucntion to use to convert alpha_ij to probabilities
                (i.e. weights that sum to 1).
        Returns:
            None
        """

        super(ConfAttention, self).__init__()

        """
        Xavier initializations from
        https://github.com/Diego999/pyGAT/blob/master/layers.py
        """

        # if boltz_basis = None then don't embed the boltmzann vector
        if boltz_basis is None:
            self.embed_boltz = False

        # otherwise initialize a layer to embed it and `fp_linear` to
        # convert cat([mol_fp, boltz_fp]) to a new vector of dimension
        # `mol_basis`.

        else:
            self.embed_boltz = True
            self.boltz_lin = torch.nn.Linear(1, boltz_basis)
            self.boltz_act = Softmax(dim=1)

            self.fp_linear = torch.nn.Linear(
                mol_basis + boltz_basis, mol_basis, bias=False)

        self.equal_weights = equal_weights

        # if you don't want equal weights, then initialize `att_weight`,
        # which will be used to get the weight og each conformer.

        if not self.equal_weights:
            self.att_weight = torch.nn.Parameter(torch.rand(1, 2 * mol_basis))
            nn.init.xavier_uniform_(self.att_weight, gain=1.414)

        # linear layer
        self.W = torch.nn.Linear(in_features=mol_basis,
                                 out_features=mol_basis)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.final_act = layer_types[final_act]()
        self.activation = LeakyReLU(inplace=False)
        self.prob_func = prob_func

    def forward(self,
                conf_fps,
                boltzmann_weights):
        """
        Args:
            conf_fps (torch.Tensor): conformer fingerprints for a species
            boltzmann_weights (torch.Tensor): their statistical Boltzmann
                weights.
        Returns:
            output (torch.Tensor): attention-pooled conformer fingerprints
            alpha_ij (torch.Tensor): the learned weight factors between
                each pair of conformers i and j.
        """

        # for backwards compatibility
        if not hasattr(self, "embed_boltz"):
            self.embed_boltz = True
        if not hasattr(self, "equal_weights"):
            self.equal_weights = False
        if not hasattr(self, "prob_func"):
            self.prob_func = 'softmax'

        if self.embed_boltz:
            # increase dimensionality of Boltzmann weight
            boltz_vec = self.boltz_act(self.boltz_lin(boltzmann_weights))

            # concatenate fingerprints with Boltzmann vector
            # and apply linear layer to reduce back to size `mol_basis`
            cat_fps = torch.cat([conf_fps, boltz_vec], dim=1)
            new_fps = self.fp_linear(cat_fps)
        else:
            new_fps = conf_fps

        # directed "neighbor list" that links every fingerprint to each other
        a = torch.LongTensor([[i, j] for i in range(new_fps.shape[0])
                              for j in range(new_fps.shape[0])]
                             ).to(conf_fps.device)

        # make cat(h_i, h_j)
        cat_ij = torch.cat((self.W(new_fps[a[:, 0]]),
                            self.W(new_fps[a[:, 1]])), dim=1)

        # number of conformers
        n_confs = new_fps.shape[0]

        # if equal weights then set all alpha_ij equal
        if self.equal_weights:
            alpha_ij = torch.ones(n_confs * n_confs, 1).to(new_fps.device)
            alpha_ij /= alpha_ij.sum()

        # otherwise proceed with regular attention

        else:
            output = self.activation(
                torch.matmul(
                    self.att_weight, cat_ij.transpose(0, 1)
                )
            )

            if self.prob_func == 'softmax':
                alpha_ij = softmax(output.reshape(n_confs, n_confs),
                                   dim=1).reshape(-1, 1) / n_confs

            elif self.prob_func == 'square':
                out_reshape = (output ** 2).reshape(n_confs, n_confs)
                norm = out_reshape.sum(dim=1).reshape(-1, 1)
                non_norm_prob = out_reshape / norm
                alpha_ij = non_norm_prob.reshape(-1, 1) / n_confs

        fp_j = new_fps[a[:, 1]]
        prod = alpha_ij * self.W(fp_j)

        # sum over neighbors and over fingerprints
        summed_fps = prod.sum(0)

        # put through nonlinearity
        output = self.final_act(summed_fps)

        return output, alpha_ij


class LinearConfAttention(ConfAttention):
    """
    Similar to ConfAttention, but instead of concatenating each pair
    of fingerprints to learn the importance of one to the other, 
    just uses each fingerprint alone to learn its importance.
    """

    def __init__(self,
                 mol_basis,
                 boltz_basis,
                 final_act,
                 equal_weights=False,
                 prob_func='softmax'):
        """
        Args:
            mol_basis (int): dimension of the molecular fingerprint
            boltz_basis (int): dimension into which we embed the boltzmann
                weight as a vector.
            final_act (str): name of the final nonlinear layer to apply to
                the fingerprint.
            equal_weights (bool): whether to not use attention and just use
                equal weights for each conformer.
            prob_func (str): fucntion to use to convert alpha_ij to probabilities
                (i.e. weights that sum to 1).
        Returns:
            None
        """

        super(LinearConfAttention, self).__init__(mol_basis,
                                                  boltz_basis,
                                                  final_act,
                                                  equal_weights)

        # has dimension mol_basis instead of 2 * mol_basis because we're not
        # comparing fingerprint pairs

        if not self.equal_weights:
            self.att_weight = torch.nn.Parameter(torch.rand(1, mol_basis))
            nn.init.xavier_uniform_(self.att_weight, gain=1.414)

        self.prob_func = prob_func

    def forward(self,
                conf_fps,
                boltzmann_weights):
        """
        Args:
            conf_fps (torch.Tensor): conformer fingerprints for a species
            boltzmann_weights (torch.Tensor): their statistical Boltzmann
                weights.
        Returns:
            output (torch.Tensor): attention-pooled conformer fingerprints
            alpha_i (torch.Tensor): the learned weight factors for
                each conformer i.
        """

        # for backwards compatibility
        if not hasattr(self, "embed_boltz"):
            self.embed_boltz = True
        if not hasattr(self, "equal_weights"):
            self.equal_weights = False
        if not hasattr(self, "prob_func"):
            self.prob_func = 'softmax'

        if self.embed_boltz:
            # increase dimensionality of Boltzmann weight
            boltz_vec = self.boltz_act(self.boltz_lin(boltzmann_weights))

            # concatenate fingerprints with Boltzmann vector
            # and apply linear layer to reduce back to size `mol_basis`
            cat_fps = torch.cat([conf_fps, boltz_vec], dim=1)
            new_fps = self.fp_linear(cat_fps)
        else:
            new_fps = conf_fps

        n_confs = new_fps.shape[0]

        # if equal weights then set all alpha_ij equal

        if self.equal_weights:
            alpha_i = torch.ones(n_confs).to(new_fps.device).reshape(-1, 1)
            alpha_i /= alpha_i.sum()

        # otherwise apply attention to each fingerprint individually

        else:
            output = self.activation(
                torch.matmul(
                    self.att_weight, self.W(new_fps).transpose(0, 1)
                )
            )

            if self.prob_func == 'softmax':
                alpha_i = softmax(output, dim=1).reshape(-1, 1)
            elif self.prob_func == 'square':
                alpha_i = (output ** 2 / (output ** 2).sum()).reshape(-1, 1)

        prod = alpha_i * self.W(new_fps)

        # sum over neighbors and over fingerprints
        summed_fps = prod.sum(0)

        # put through nonlinearity
        output = self.final_act(summed_fps)

        return output, alpha_i


class ChemPropConv(MessagePassingModule):

    """
    ChemProp convolution module.
    """

    def __init__(self,
                 n_edge_hidden,
                 dropout_rate,
                 activation,
                 **kwargs):
        """
        Args:
            n_edge_hidden: dimension of the hidden edge vector
            dropout_rate (float): dropout rate
            activation (str): name of non-linear activation function
        Returns:
            None
        """

        MessagePassingModule.__init__(self)

        # As in the original ChemProp paper,
        # the features are added together linearly
        # with no bias. This means that features
        # equal to 0 don't contribute to the output.

        # This is important, for example, for
        # CpSchNetConv, in which every non-
        # bonded neighbour has zeros for its
        # ChemProp bond features. We don't want
        # these zeros contributing to the output.

        self.dense = Dense(
            in_features=n_edge_hidden,
            out_features=n_edge_hidden,
            dropout_rate=dropout_rate,
            bias=False
        )
        self.activation = layer_types[activation]()

    def message(self,
                h_new,
                nbrs,
                ji_idx=None,
                kj_idx=None):
        """
        Get the chemprop MPNN message.
        Args:
            h_new (torch.Tensor): new hidden edge vector
            nbrs (torch.LongTensor): bonded neighbor list
            ji_idx (torch.LongTensor, optional): a set of indices for the neighbor list
            kj_idx (torch.LongTensor, optional): a set of indices for the neighbor list
                such that nbrs[kj_idx[n]][0] == nbrs[ji_idx[n]][1] for any
                value of n.
        Returns:
            msg (torch.Tensor): message from nearby atoms
        """

        msg = chemprop_msg_update(h=h_new,
                                  nbrs=nbrs,
                                  ji_idx=ji_idx,
                                  kj_idx=kj_idx)
        return msg

    def update(self, msg, h_0):
        """
        Update the hiddn edge features.
        Args:
            msg (torch.Tensor): message vector
            h_0 (torch.Tensor): initial hidden edge
                vector.
        Returns:
            update_feats (torch.Tensor): updated
                features after convolution.
        """

        # addition of original edge features with
        # `msg` after going through a dense layer
        add_feats = h_0 + self.dense(msg)
        # apply the nonlinearity
        update_feats = self.activation(add_feats)

        return update_feats

    def forward(self,
                h_0,
                h_new,
                nbrs,
                ji_idx=None,
                kj_idx=None):
        """
        Apply a convolution layer.
        Args:
            h_0 (torch.Tensor): initial hidden edge
                vector.
            h_new (torch.Tensor): latest hidden edge vector
            nbrs (torch.LongTensor): bonded neighbor list
            ji_idx (torch.LongTensor, optional): a set of indices for the neighbor list
            kj_idx (torch.LongTensor, optional): a set of indices for the neighbor list
                such that nbrs[kj_idx[n]][0] == nbrs[ji_idx[n]][1] for any
                value of n.
        Returns:
            update_feats (torch.Tensor): updated
                features after convolution.
        """

        msg = self.message(h_new=h_new,
                           nbrs=nbrs,
                           ji_idx=ji_idx,
                           kj_idx=kj_idx)
        update_feats = self.update(msg=msg,
                                   h_0=h_0)

        return update_feats


class CpSchNetConv(ChemPropConv):
    """
    Module for combining a ChemProp convolution with non-convolved
    distance features. 
    """

    def __init__(
        self,
        n_bond_hidden,
        cp_dropout,
        gauss_embed,
        cutoff,
        n_gaussians,
        trainable_gauss,
        n_filters,
        schnet_dropout,
        activation,
        **kwargs

    ):
        """
        Args:
            n_bond_hidden (int): bond feature hidden dimension
            cp_dropout (float): dropout rate for the ChemProp convolution
            gauss_embed (bool): whether to embed distances in a
                basis of Gaussians.
            cutoff (float): neighbor list cutoff
            n_gaussians (int): number of Gaussians in which to expand
                distances.
            trainable_gauss (bool): whether Gaussian spacings and widths
                are learnable parameters.
            n_filters (int): hidden distance feature dimension
            schnet_dropout (float): dropout rate for SchNet embedding
            activation (str): name of nonlinear activation function
        Returns:
            None
        """

        ChemPropConv.__init__(self,
                              n_edge_hidden=n_bond_hidden,
                              dropout_rate=cp_dropout,
                              activation=activation)

        self.n_bond_hidden = n_bond_hidden
        self.moduledict = ModuleDict({})

        if not gauss_embed:
            return

        edge_filter = Sequential(
            GaussianSmearing(
                start=0.0,
                stop=cutoff,
                n_gaussians=n_gaussians,
                trainable=trainable_gauss,
            ),
            Dense(
                in_features=n_gaussians,
                out_features=n_filters,
                dropout_rate=schnet_dropout,
            ),
            layer_types[activation]())

        self.moduledict["edge_filter"] = edge_filter

    def add_schnet_feats(self, e, h_new):
        """
        Add distance features to the ChemProp updated bond
        features.
        Args:
            e (torch.Tensor): distance features
            h_new (torch.Tensor): updated bond features 
        Returns:
            new_msg (torch.Tensor): concatenation of
                bond and distance edge features.
        """

        if "edge_filter" in self.moduledict:
            e = self.moduledict["edge_filter"](e)

        new_msg = torch.cat((h_new, e), dim=1)

        return new_msg

    def forward(self,
                h_0,
                h_new,
                all_nbrs,
                bond_nbrs,
                bond_idx,
                e,
                ji_idx=None,
                kj_idx=None):
        """
        Update the edge features.
        Args:
            h_0 (torch.Tensor): original edge features
            h_new (torch.Tensor): latest updated version
                of edge features.
            all_nbrs (torch.LongTensor): full neighbor list
            bond_nbrs (torch.LongTensor): bonded neighbor
                list
            bond_idx (torch.LongTensor): a list that maps a bonded
                pair to the corresponding index in the neighbor list.
            e (torch.Tensor): distances between atoms
            ji_idx (torch.LongTensor, optional): a set of indices for the neighbor list
            kj_idx (torch.LongTensor, optional): a set of indices for the neighbor list
                such that nbrs[kj_idx[n]][0] == nbrs[ji_idx[n]][1] for any
                value of n.

        Returns:
            final_h (torch.Tensor): new updated version of edge features
        """

        # `h_new` is a concatenation of the ChemProp hidden vector h
        # with the distance features. So we slice from 0 to self.n_bond_hidden,
        # then select only the non-zero indices at `bond_idx`, to get the
        # latest ChemProp edge vector

        cp_h = h_new[:, :self.n_bond_hidden][bond_idx]

        # `h_0` is only a ChemProp vector, but padded with zeros
        h0_bond = h_0[bond_idx]

        # get the ChemProp message and update the ChemProp `h`
        cp_msg = self.message(h_new=cp_h,
                              nbrs=bond_nbrs,
                              ji_idx=ji_idx,
                              kj_idx=kj_idx)

        h_new_bond = self.update(msg=cp_msg,
                                 h_0=h0_bond)

        # pad it back with zeros for non-bonded atoms
        nbr_dim = all_nbrs.shape[0]
        h_new = torch.zeros((nbr_dim,  self.n_bond_hidden))
        h_new = h_new.to(bond_idx.device)
        h_new[bond_idx] = h_new_bond

        # concatenate with SchNet distance features
        final_h = self.add_schnet_feats(e=e,
                                        h_new=h_new)

        return final_h


class ChemPropMsgToNode(nn.Module):
    """
    Convert ChemProp edge features to message features.
    """

    def __init__(self, output_layers):
        """
        Args:
            output_layers (list[dict]): instructions for
                making the output layers applied after the
                initial node features get concatenated with
                the edge-turned-node updated features.
        Returns:
            None 
        """
        nn.Module.__init__(self)

        # remove bias from linear layers if there

        new_layers = remove_bias(output_layers)
        self.output = construct_sequential(new_layers)

    def forward(self, r, h, nbrs):
        """
        Call the module.
        Args:
            r (torch.Tensor): initial node features
            h (torch.Tensor): latest edge features
            nbrs (torch.LongTensor): neighbor list
        Returns:
            new_node_feats (torch.Tensor): latest node
                features.
        """
        num_nodes = r.shape[0]
        msg_to_node = chemprop_msg_to_node(h=h,
                                           nbrs=nbrs,
                                           num_nodes=num_nodes)
        cat_node = torch.cat((r, msg_to_node), dim=1)
        new_node_feats = self.output(cat_node)

        return new_node_feats


class ChemPropInit(nn.Module):
    """
    Initial module that converts node and edge features
    to hidden edge features in ChemProp.
    """

    def __init__(self, input_layers):
        """
        Args:
            input_layers (list[dict]): instructions for
                making the input layers applied to the node
                and edge features.
        Returns:
            None 
        """
        nn.Module.__init__(self)

        # remove bias from linear layers if there

        new_layers = remove_bias(input_layers)
        self.input = construct_sequential(new_layers)

    def forward(self, r, bond_feats, bond_nbrs):
        """
        Call the module.
        Args:
            r (torch.Tensor): initial node features
            bond_feats (torch.Tensor): initial bond features
            bond_nbrs (torch.LongTensor): bonded neighbor list
        Returns:
            hidden_feats (torch.Tensor): hidden edge features
        """
        cat_feats = torch.cat((r[bond_nbrs[:, 0]], bond_feats),
                              dim=1)
        hidden_feats = self.input(cat_feats)

        return hidden_feats


class DiabaticReadout(nn.Module):
    def __init__(self,
                 diabat_keys,
                 grad_keys,
                 energy_keys,
                 delta=False,
                 stochastic_dic=None):

        nn.Module.__init__(self)

        self.diag = Diagonalize()
        self.diabat_keys = diabat_keys
        self.grad_keys = grad_keys
        self.energy_keys = energy_keys
        self.delta = delta
        self.stochastic_modules = self.make_stochastic(stochastic_dic)
        # self.sigma_delta_keys = sigma_delta_keys

    def make_stochastic(self, stochastic_dic):
        """
        E.g. stochastic_layers = {"energy_1": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.5},
                                "lam": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.25}
                                    },
                                 "d1": 
                                    {"name": "stochasticincrease",
                                    "param": {"exp_coef": 3,
                                             "order": 4,
                                             "rate": 0.5}}

        For energy_1 it's understood that the adiabatic gap between state 1
        and state 0 will be increased. Similarly for d1 is's understood that
        the diabatic gap between state 1 and state 0 will be increased. If we
        had also specified, for example, energy_2 and d2, then those gaps
        will be increased relative to the new values of energy_1 and d_1.

        For "lam", an off-diagonal diabatic element, it's understood only that
        its magnitude will decrease.
        """

        stochastic_modules = nn.ModuleDict({})
        if stochastic_dic is None:
            return stochastic_modules

        for key, layer_dic in stochastic_dic.items():
            if layer_dic["name"].lower() == "stochasticincrease":
                params = layer_dic["param"]
                layer = StochasticIncrease(**params)
                stochastic_modules[key] = layer
            else:
                raise NotImplementedError

        return stochastic_modules

    def get_nacv(self, U, xyz, N):

        num_states = U.shape[2]
        split_xyz = torch.split(xyz, N)

        u_grads = [torch.zeros(num_states, num_states,
                               this_xyz.shape[0], 3).to(xyz.device)
                   for this_xyz in split_xyz]

        for i in range(U.shape[1]):
            for j in range(U.shape[2]):
                if i == j:
                    continue
                this_grad = compute_grad(inputs=xyz,
                                         output=U[:, i, j]).detach()
                grad_split = torch.split(this_grad, N)
                for k, grad in enumerate(grad_split):
                    u_grads[k][i, j, :, :] = grad

        U = U.detach()

        # m, l, and s are state indices that get summed out
        # i and j are state indices that don't get summed out
        # a = N_at is the number of atoms
        # t = 3 is the number of directions for each atom

        nacvs = []
        for k, u_grad in enumerate(u_grads):
            this_u = U[k]
            nacv = torch.einsum('im, sjat, lm, sl -> ijat',
                                this_u, u_grad, this_u, this_u)
            nacvs.append(nacv)

        # concatenate along all the atoms just like we do
        # for forces, giving a tensor of shape
        # (num_states, num_states, total_num_atoms, 3)

        nacv_cat = torch.cat(nacvs, axis=-2)

        return nacv_cat

    def add_nacv(self, results, u, xyz, N):

        nacv = self.get_nacv(U=u, xyz=xyz, N=N)
        num_states = nacv.shape[0]
        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    continue
                this_nacv = nacv[i, j, :, :]
                results[f"nacv_{i}{j}"] = this_nacv
        return results

    def add_diag(self,
                 results,
                 N,
                 xyz,
                 add_nacv):

        diabat_keys = np.array(self.diabat_keys)
        dim = diabat_keys.shape[0]
        num_geoms = len(N)
        diabat_ham = (torch.zeros(num_geoms, dim, dim)
                      .to(xyz.device))
        for i in range(dim):
            for j in range(dim):
                key = diabat_keys[i, j]
                diabat_ham[:, i, j] = results[key]

        ad_energies, u = self.diag(diabat_ham)

        results.update({key: ad_energies[:, i].reshape(-1, 1)
                        for i, key in enumerate(self.energy_keys)})
        if add_nacv:
            results = self.add_nacv(results=results,
                                    u=u,
                                    xyz=xyz,
                                    N=N)

        return results

    def choose_grad_route(self, extra_grads):
        """
        If gradients of certain diabatic states are asked for, then
        decide the most efficient way to calculate both those and
        the adiabatic gradients.
        """

        # unique diabatic quantities
        unique_diabats = list(set(np.array(self.diabat_keys)
                                  .reshape(-1).tolist()))

        # extra quantities whose gradients were requested
        extra_quants = [i.replace("_grad", "") for i in
                        np.array(extra_grads).reshape(-1).tolist()]

        # diabatic keys for which gradients were requested
        diabats_need_grad = list(set([i for i in extra_quants
                                      if i in unique_diabats]))

        # adiabatic energies for which gradients were requested
        energies_need_grad = [i.replace("_grad", "") for i in self.grad_keys
                              if i.replace("_grad", "") in self.energy_keys]

        # number of diabatic gradients needed to make sure we get all the
        # adiabatic gradients right

        num_diabat_to_en = len(unique_diabats)

        # number of gradients needed if we compute all gradients separately

        num_separate = len(diabats_need_grad) + len(energies_need_grad)

        # choose the route that takes fewer calculations

        if num_diabat_to_en < num_separate:
            route = "diabat_to_en"
        else:
            route = "separate"

        return route

    def compute_diabat_grad(self,
                            results,
                            xyz,
                            N):
        unique_diabats = list(set(np.array(self.diabat_keys)
                                  .reshape(-1).tolist()))

        grad_dic = {}
        for d_key in unique_diabats:
            grad = compute_grad(inputs=xyz, output=results[d_key])
            grad_dic[f"{d_key}_grad"] = grad

        # num_diabat x num_atoms x 3
        d_grads = torch.stack([grad_dic[d_key + "_grad"]
                               for d_key in unique_diabats])

        return grad_dic, d_grads, unique_diabats

    def compute_dE_dD(self,
                      results,
                      en_keys,
                      unique_diabats,
                      num_mols,
                      num_states):

        num_diabat = len(unique_diabats)
        device = results[en_keys[0]].device

        # num_mols x num_states x num_diabat
        dE_dD = torch.zeros(num_mols, num_states, num_diabat)

        for i, en_key in enumerate(en_keys):
            for j, d_key in enumerate(unique_diabats):
                grad = compute_grad(inputs=results[d_key],
                                    output=results[en_key])
                dE_dD[:, i, j] = grad

        return dE_dD

    def compute_all_grads(self,
                          results,
                          xyz,
                          N):
        """
        Compute gradients of all diabatic energies and then
        of the adiabatic energies requested.
        """

        en_keys = [i.replace("_grad", "") for i in self.grad_keys
                   if i.replace("_grad", "") in self.energy_keys]
        num_states = len(en_keys)
        num_mols = results[en_keys[0]].shape[0]

        # d_grads: num_diabat x num_atoms x 3
        grad_dic, d_grads, unique_diabats = self.compute_diabat_grad(
            results=results,
            xyz=xyz,
            N=N)

        # dE_dD: num_mols x num_states x num_diabat
        dE_dD = self.compute_dE_dD(results=results,
                                   en_keys=en_keys,
                                   unique_diabats=unique_diabats,
                                   num_mols=num_mols,
                                   num_states=num_states)

        # do molecule by molecule
        num_atoms = d_grads.shape[1]
        mol_d_grads = torch.split(d_grads, N, dim=1)
        all_engrads = torch.zeros(num_states, num_atoms, 3)

        counter = 0

        for i in range(num_mols):
            # num_diabat x (num_atoms of this mol) x 3
            mol_d_grad = mol_d_grads[i]

            # num_states x num_diabat
            mol_dE_dD = dE_dD[i].to(mol_d_grad.device)

            # output = num_states x (num_atoms of this_mol) x 3
            engrads = torch.einsum("ij,jkl->ikl", mol_dE_dD, mol_d_grad)

            # put into concatenated gradients
            this_num_atoms = mol_d_grad.shape[1]
            all_engrads[:, counter: counter + this_num_atoms, :] = engrads

            counter += this_num_atoms

        for j, en_key in enumerate(en_keys):
            grad_dic[en_key + "_grad"] = all_engrads[j]

        return grad_dic

    def add_grad(self,
                 results,
                 xyz,
                 N,
                 extra_grads=None,
                 try_speedup=False):

        # for example, if you want the gradients of the diabatic
        # energies

        if extra_grads is not None:

            # For two states you can get a speed-up by first
            # computing the gradients of all diabatic quantities
            # and then using the chain rule to get the adiabatic
            # gradients. This slows things down for >= 4 states
            # if you only want diagonal diabatic gradients.
            # The function `choose_grad_route` identifies which
            # method should be faster.

            # This provides a speedup on cpu but actually slows
            # things down on GPU, possibly because of having
            # to move dE_dD to the GPU. The increase in time
            # is actually only about 25% when doing in the
            # naive way for a batch size of 20 and 2 states.

            grad_route = self.choose_grad_route(extra_grads)

            if try_speedup and grad_route == "diabat_to_en":
                grads = self.compute_all_grads(results, xyz, N)
                results.update(grads)
                return results

            all_grad_keys = [*self.grad_keys, *extra_grads]
        else:
            all_grad_keys = self.grad_keys

        # import pdb
        # pdb.set_trace()

        for grad_key in all_grad_keys:
            if "_grad" not in grad_key:
                grad_key += "_grad"

            base_key = grad_key.replace("_grad", "")
            output = results[base_key]

            grad = compute_grad(inputs=xyz, output=output)

            results[grad_key] = grad

        return results

    def add_gap(self, results):

        # diabatic gap

        bottom_key = self.diabat_keys[0][0]
        top_key = self.diabat_keys[1][1]
        gap = results[top_key] - results[bottom_key]
        results.update({"abs_diabat_gap": abs(gap)})

        # adiabatic gap

        num_states = len(self.energy_keys)
        for i in range(num_states):
            for j in range(num_states):
                if j <= i:
                    continue

                upper_key = self.energy_keys[j]
                lower_key = self.energy_keys[i]
                gap = results[upper_key] - results[lower_key]
                results.update({f"{upper_key}_{lower_key}_delta": gap})

        return results

    def add_stochastic(self, results):

        # any deltas that you want to decrease, whether adiabatic
        # or diabatic

        module_keys = self.stochastic_modules.keys()
        diabat_keys = np.array(self.diabat_keys)
        diag_diabat_keys = np.array(self.diabat_keys).diagonal()
        num_states = diabat_keys.shape[0]

        diag_adiabat_incr = [i for i in module_keys if i
                             not in diabat_keys.reshape(-1)]
        odiag_diabat_incr = []
        diag_diabat_incr = []

        for i in range(num_states):
            for j in range(num_states):
                key = diabat_keys[i, j]
                if key not in module_keys:
                    continue
                if i == j:
                    diag_diabat_incr.append(key)
                else:
                    odiag_diabat_incr.append(key)

        odiag_diabat_incr = list(set(odiag_diabat_incr))
        # all diagonals, both adiabatic and diabatic
        all_diag_incr = list(set(diag_adiabat_incr + diag_diabat_incr))

        # directly scale off-diagonal diabatic elements
        for key in odiag_diabat_incr:
            output = results[key]
            results[key] = self.stochastic_modules[key](output)

        # sequentially increase gap between adiabatic and diagonal
        # diabatic elements

        for diag_keys in [diag_diabat_keys, self.energy_keys]:
            for i, key in enumerate(diag_keys):
                # start with first excited state, meaning you ignore
                # the lowest key
                if i == 0:
                    continue
                # don't do anything if we didn't ask it to be scaled
                if key not in all_diag_incr:
                    continue

                lower_key = diag_diabat_keys[i - 1]
                delta = results[key] - results[lower_key]

                # stochastically increase the difference between the
                # two states
                change = -delta + self.stochastic_modules[key](delta)
                results[key] = results[key] + change

        return results

    def forward(self,
                batch,
                xyz,
                results,
                add_nacv=False,
                add_grad=True,
                add_gap=True,
                extra_grads=None,
                try_speedup=False):

        if not hasattr(self, "delta"):
            self.delta = False

        if not hasattr(self, "stochastic_modules"):
            self.stochastic_modules = nn.ModuleDict({})

        # if not hasattr(self, "sigma_delta_keys"):
        #     self.sigma_delta_keys = None

        N = batch["num_atoms"].detach().cpu().tolist()

        # must go before computing the diagonals!

        if self.delta:
            diag_diabat = np.array(self.diabat_keys).diagonal()
            for key in diag_diabat[1:]:
                results[key] += results[diag_diabat[0]]

            # diag_diabat = np.array(self.diabat_keys).diagonal()
            # assert diag_diabat.shape[0] == 2

            # # d0 = sigma - delta, d1 = sigma + delta

            # key_0 = diag_diabat[0]
            # key_1 = diag_diabat[1]

            # sigma_key = self.sigma_delta_keys[0]
            # delta_key = self.sigma_delta_keys[1]

            # results[key_0] = results[sigma_key] - results[delta_key]
            # results[key_1] = results[sigma_key] + results[delta_key]

        results = self.add_diag(results=results,
                                N=N,
                                xyz=xyz,
                                add_nacv=add_nacv)

        if add_grad:
            results = self.add_grad(results=results,
                                    xyz=xyz,
                                    N=N,
                                    extra_grads=extra_grads,
                                    try_speedup=try_speedup)

        if add_gap:
            results = self.add_gap(results)

        if self.training:
            results = self.add_stochastic(results)

        return results


def sum_and_grad(batch,
                 xyz,
                 atomwise_output,
                 grad_keys,
                 out_keys=None):

    N = batch["num_atoms"].detach().cpu().tolist()
    results = {}
    if out_keys is None:
        out_keys = list(atomwise_output.keys())

    for key, val in atomwise_output.items():
        if key not in out_keys:
            continue
        # split the outputs into those of each molecule
        split_val = torch.split(val, N)
        # sum the results for each molecule
        results[key] = torch.stack([i.sum() for i in split_val])

    # compute gradients

    for key in grad_keys:
        output = results[key.replace("_grad", "")]
        grad = compute_grad(output=output,
                            inputs=xyz)
        results[key] = grad

    return results


class SumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                batch,
                xyz,
                atomwise_output,
                grad_keys,
                out_keys=None):
        results = sum_and_grad(batch=batch,
                               xyz=xyz,
                               atomwise_output=atomwise_output,
                               grad_keys=grad_keys,
                               out_keys=out_keys)
        return results


def att_readout_probs(name):
    if name.lower() == "softmax":
        def func(output):
            weights = softmax(output, dim=0)
            return weights

    elif name.lower() == "square":
        def func(output):
            weights = ((output ** 2 / (output ** 2).sum()))
            return weights
    else:
        raise NotImplementedError

    return func


class AttentionPool(nn.Module):
    """
    Compute output quantities using attention, rather than a sum over
    atomic quantities. There are two methods to do this:
    (1): "atomwise": Learn the attention weights from atomic fingerprints,
    get atomwise quantities from a network applied to the fingeprints,
    and sum them with attention weights.
    (2) "mol_fp": Learn the attention weights from atomic fingerprints,
    multiply the fingerprints by these weights, add the fingerprints
    together to get a molecular fingerprint, and put the molecular
    fingerprint through a network that predicts the output.

    This one uses `mol_fp`, since it seems more expressive (?)
    """

    def __init__(self,
                 prob_func,
                 feat_dim,
                 att_act,
                 mol_fp_act,
                 num_out_layers,
                 out_dim):
        """

        """
        super().__init__()

        self.w_mat = nn.Linear(in_features=feat_dim,
                               out_features=feat_dim,
                               bias=False)

        self.att_weight = torch.nn.Parameter(torch.rand(1, feat_dim))
        nn.init.xavier_uniform_(self.att_weight, gain=1.414)
        self.prob_func = att_readout_probs(prob_func)
        self.att_act = layer_types[att_act]()

        # reduce the number of features by the same factor in each layer
        feat_num = [int(feat_dim / num_out_layers ** m)
                    for m in range(num_out_layers)]

        # make layers followed by an activation for all but the last
        # layer
        mol_fp_layers = [Dense(in_features=feat_num[i],
                               out_features=feat_num[i+1],
                               activation=layer_types[mol_fp_act]())
                         for i in range(num_out_layers - 1)]

        # use no activation for the last layer
        mol_fp_layers.append(Dense(in_features=feat_num[-1],
                                   out_features=out_dim,
                                   activation=None))

        # put together in readout network
        self.mol_fp_nn = Sequential(*mol_fp_layers)

    def forward(self,
                batch,
                xyz,
                atomwise_output,
                grad_keys,
                out_keys):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key in out_keys:

            # batched_feats = atomwise_output[key]
            batched_feats = atomwise_output['features']

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []

            for feats in split_feats:
                weights = self.prob_func(
                    self.att_act(
                        (self.att_weight * self.w_mat(feats)).sum(-1)
                    )
                )

                mol_fp = (weights.reshape(-1, 1) * self.w_mat(feats)).sum(0)
                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)

            results[key] = torch.stack(all_outputs).reshape(-1)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results


class ScaleShift(nn.Module):

    r"""Scale and shift layer for standardization.
    .. math::
       y = x \times \sigma + \mu
    Args:
        means (dict): dictionary of mean values
        stddev (dict): dictionary of standard deviations
    """

    def __init__(self,
                 means=None,
                 stddevs=None):
        super(ScaleShift, self).__init__()

        means = means if (means is not None) else {}
        stddevs = stddevs if (stddevs is not None) else {}
        self.means = means
        self.stddevs = stddevs

    def forward(self, inp, key):
        """Compute layer output.
        Args:
            inp (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """

        stddev = self.stddevs.get(key, 1.0)
        mean = self.means.get(key, 0.0)
        out = inp * stddev + mean

        return out


class TestModules(unittest.TestCase):
    def testBaseEdgeUpdate(self):
        # initialize basic graphs

        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e = torch.rand(5, 10)
        r_in = torch.rand(6, 10)
        model = MessagePassingModule()
        r_out = model(r_in, e, a)
        self.assertEqual(
            r_in.shape,
            r_out.shape,
            "The node feature dimensions should be same for the base case",
        )

    def testBaseMessagePassing(self):
        # initialize basic graphs
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e_in = torch.rand(5, 10)
        r = torch.rand(6, 10)
        model = EdgeUpdateModule()
        e_out = model(r, e_in, a)
        self.assertEqual(
            e_in.shape,
            e_out.shape,
            "The edge feature dimensions should be same for the base case",
        )

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

        self.assertEqual(
            r_in.shape, r_out.shape,
            "The node feature dimensions should be same."
        )

    def testDoubleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers = [
            {
                "name": "linear",
                "param": {
                    "in_features": 2 * num_features,
                    "out_features": num_features,
                },
            },
            {"name": "shifted_softplus", "param": {}},
            {
                "name": "linear",
                "param": {"in_features": num_features,
                          "out_features": num_features},
            },
            {"name": "shifted_softplus", "param": {}},
        ]

        r_in = torch.rand(num_nodes, num_features)
        model = DoubleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

        self.assertEqual(
            r_in.shape, r_out.shape,
            "The node feature dimensions should be same."
        )

    def testSingleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers = [
            {
                "name": "linear",
                "param": {"in_features": num_features,
                          "out_features": num_features},
            },
            {"name": "shifted_softplus", "param": {}},
            {
                "name": "linear",
                "param": {"in_features": num_features,
                          "out_features": num_features},
            },
            {"name": "shifted_softplus", "param": {}},
        ]

        r_in = torch.rand(num_nodes, num_features)
        model = SingleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

        self.assertEqual(
            r_in.shape,
            r_out.shape,
            ("The node feature dimensions should be same for the "
             "SchNet Convolution case"),
        )

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

        self.assertEqual(
            e_in.shape,
            e_out.shape,
            ("The edge feature dimensions should be same for the SchNet "
             "Edge Update case"),
        )

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
            "myenergy0": [
                {"name": "Dense", "param": {"in_features": 5,
                                            "out_features": 20}},
                {"name": "shifted_softplus", "param": {}},
                {"name": "Dense", "param": {"in_features": 20,
                                            "out_features": 1}},
            ],
            "myenergy1": [
                {"name": "linear", "param": {"in_features": 5,
                                             "out_features": 20}},
                {"name": "Dense", "param": {"in_features": 20,
                                            "out_features": 1}},
            ],
            "Muliken charges": [
                {"name": "linear", "param": {"in_features": 5,
                                             "out_features": 20}},
                {"name": "linear", "param": {"in_features": 20,
                                             "out_features": 1}},
            ],
        }

        model = NodeMultiTaskReadOut(multitaskdict)
        output = model(r)


if __name__ == "__main__":
    unittest.main()
