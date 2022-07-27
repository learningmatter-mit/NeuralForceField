
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict
from torch.nn import Softmax
from torch.nn.functional import softmax

import unittest

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add, compute_grad
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import (
    MessagePassingModule,
    EdgeUpdateModule,
)
from nff.nn.utils import (construct_sequential, construct_module_dict,
                          chemprop_msg_update, chemprop_msg_to_node,
                          remove_bias)
from nff.utils.tools import layer_types

# for backwards compatability
from nff.nn.modules.diabat import DiabaticReadout

EPSILON = 1e-15
DEFAULT_BONDPRIOR_PARAM = {"k": 20.0}


def get_offsets(batch, key):
    nxyz = batch['nxyz']
    zero = torch.Tensor([0]).to(nxyz.device)
    offsets = batch.get(key, zero)
    if isinstance(offsets, torch.Tensor) and offsets.is_sparse:
        offsets = offsets.to_dense()
    return offsets


def get_rij(xyz,
            batch,
            nbrs,
            cutoff):

    offsets = get_offsets(batch, 'offsets')
    # + offsets not - offsets because it's r_j - r_i,
    # whereas for schnet we've coded it as r_i - r_j
    r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]] + offsets

    # remove nbr skin (extra distance added to cutoff
    # to catch atoms that become neighbors between nbr
    # list updates)
    dist = (r_ij.detach() ** 2).sum(-1) ** 0.5

    if type(cutoff) == torch.Tensor:
        dist = dist.to(cutoff.device)
    use_nbrs = (dist <= cutoff)

    r_ij = r_ij[use_nbrs]
    nbrs = nbrs[use_nbrs]

    return r_ij, nbrs


def add_stress(batch,
               all_results,
               nbrs,
               r_ij):
    """
    Add stress as output. Needs to be divided by lattice volume to get actual stress. 
    For batching for loop seemed unavoidable. will change later.
    stress considers both for crystal and molecules. 
    For crystals need to divide by lattice volume. 
    r_ij considers offsets which is different for molecules and crystals.
    """

    Z = compute_grad(output=all_results['energy'],
                     inputs=r_ij)
    if batch['num_atoms'].shape[0] == 1:
        all_results['stress_volume'] = torch.matmul(Z.t(), r_ij)
    else:
        allstress = []
        for j in range(batch['nxyz'].shape[0]):
            allstress.append(
                torch.matmul(
                    Z[torch.where(nbrs[:, 0] == j)].t(),
                    r_ij[torch.where(nbrs[:, 0] == j)]
                )
            )
        allstress = torch.stack(allstress)
        N = batch["num_atoms"].detach().cpu().tolist()
        split_val = torch.split(allstress, N)
        all_results['stress_volume'] = torch.stack([i.sum(0)
                                                    for i in split_val])
    return all_results


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

        # for backwards compatability
        if not hasattr(self, "readout"):
            self.readout = construct_module_dict(
                self.multitaskdict['atom_readout'])
            self.readout.to(r.device)

        for key in self.readout:
            predict_dict[key] = self.readout[key](r)

        if getattr(self, "post_readout", None) is not None:
            predict_dict = self.post_readout(predict_dict)

        ###
        # predict_dict['energy_0'] = predict_dict['d0']
        # predict_dict['energy_1'] = predict_dict['d1']
        ###

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


def sum_and_grad(batch,
                 xyz,
                 atomwise_output,
                 grad_keys,
                 out_keys=None,
                 mean=False):

    N = batch["num_atoms"].detach().cpu().tolist()
    results = {}
    if out_keys is None:
        out_keys = list(atomwise_output.keys())

    for key, val in atomwise_output.items():
        if key not in out_keys:
            continue

        mol_idx = torch.arange(len(N)).repeat_interleave(
            torch.LongTensor(N)).to(val.device)
        dim_size = mol_idx.max() + 1

        if val.reshape(-1).shape[0] == mol_idx.shape[0]:
            use_val = val.reshape(-1)

        # summed atom features
        elif val.shape[0] == mol_idx.shape[0]:
            use_val = val.sum(-1)

        else:
            raise Exception(("Don't know how to handle val shape "
                             "{} for key {}" .format(val.shape, key)))

        pooled_result = scatter_add(use_val,
                                    mol_idx,
                                    dim_size=dim_size)
        if mean:
            pooled_result = pooled_result / torch.Tensor(N).to(val.device)

        results[key] = pooled_result

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


class MeanPool(nn.Module):
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
                               out_keys=out_keys,
                               mean=True)
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
                 out_dim,
                 **kwargs):
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

            batched_feats = atomwise_output['features']

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                weights = self.prob_func(
                    self.att_act(
                        (self.att_weight * self.w_mat(feats)).sum(-1)
                    )
                )

                mol_fp = (weights.reshape(-1, 1) * self.w_mat(feats)).sum(0)

                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results


class MolFpPool(nn.Module):
    def __init__(self,
                 feat_dim,
                 mol_fp_act,
                 num_out_layers,
                 out_dim,
                 **kwargs):

        super().__init__()

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

            batched_feats = atomwise_output['features']

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                mol_fp = feats.sum(0)
                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

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


def get_act(activation):
    return layer_types[activation]()


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
