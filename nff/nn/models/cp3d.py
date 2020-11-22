from torch import nn
import torch
import numpy as np
import math
from nff.data.graphs import get_bond_idx

from nff.nn.models.conformers import WeightedConformers
from nff.nn.modules import (ChemPropConv, ChemPropMsgToNode,
                            ChemPropInit, SchNetEdgeFilter,
                            CpSchNetConv)
from nff.utils.tools import make_directed
from nff.utils.confs import split_batch

REINDEX_KEYS = ["nbr_list", "bonded_nbr_list"]


class ChemProp3D(WeightedConformers):
    """
    Model that uses a representation of a molecule in terms of different 3D
    conformers to predict properties. The fingerprints of each conformer are
    generated using a 3D extension of the ChemProp model to include distance
    information. The 3D information is featurized using a SchNet Gaussian
    filter.

    """

    def __init__(self, modelparams):
        """
        Initialize model.
        Args:
            modelparams (dict): dictionary of parameters for the model
        Returns:
            None
        """

        WeightedConformers.__init__(self, modelparams)
        # get rid of the atom embedding, as we'll be using graph-based
        # atom features instead of atomic number embeddings
        delattr(self, "atom_embed")

        cp_input_layers = modelparams["cp_input_layers"]
        schnet_input_layers = modelparams["schnet_input_layers"]
        output_layers = modelparams["output_layers"]

        # make the convolutions, the input networks W_i for both
        # SchNet and ChemProp, and the output network W_o

        self.W_i_cp = ChemPropInit(input_layers=cp_input_layers)
        self.W_i_schnet = ChemPropInit(input_layers=schnet_input_layers)
        self.convolutions = self.make_convs(modelparams)
        self.W_o = ChemPropMsgToNode(output_layers=output_layers)

        # dimension of the hidden SchNet distance edge ector
        self.n_filters = modelparams["n_filters"]

        # edge filter to convert distances to SchNet feature vectors
        self.edge_filter = SchNetEdgeFilter(
            cutoff=modelparams["cutoff"],
            n_gaussians=modelparams["n_gaussians"],
            trainable_gauss=modelparams["trainable_gauss"],
            n_filters=modelparams["n_filters"],
            dropout_rate=modelparams["dropout_rate"],
            activation=modelparams["activation"])

    def make_convs(self, modelparams):
        """
        Make the convolution layers.
        Args:
            modelparams (dict): dictionary of parameters for the model
        Returns:
            convs (nn.ModuleList): list of networks for each convolution
        """

        num_conv = modelparams["n_convolutions"]
        modelparams.update({"n_edge_hidden": modelparams["mol_basis"]})

        # call `CpSchNetConv` to make the convolution layers
        convs = nn.ModuleList([ChemPropConv(**modelparams)
                               for _ in range(num_conv)])

        return convs

    def get_distance_feats(self,
                           batch,
                           xyz,
                           offsets,
                           bond_nbrs):
        """
        Get distance features.
        Args:
            batch (dict): batched sample of species
            xyz (torch.Tensor): xyz of the batch
            offsets (float): periodic boundary condition offsets
            bond_nbrs (torch.LongTensor): directed bonded neighbor list
        Returns:
            nbr_list (torch.LongTensor): directed neighbor list
            distance_feats (torch.Tensor): distance-based edge features
            bond_idx (torch.LongTensor): indices that map bonded atom pairs
                to their location in the neighbor list.
        """

        # get directed neighbor list
        nbr_list, nbr_was_directed = make_directed(batch["nbr_list"])
        # distances
        distances = (xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]] -
                     offsets).pow(2).sum(1).sqrt()[:, None]
        # put through Gaussian filter and dense layer to get features
        distance_feats = self.edge_filter(distances)

        # get the bond indices, and adjust as necessary if the neighbor list
        # wasn't directed before

        if "bond_idx" in batch:
            bond_idx = batch["bond_idx"]
            if not nbr_was_directed:
                nbr_dim = nbr_list.shape[0]
                bond_idx = torch.cat([bond_idx,
                                      bond_idx + nbr_dim // 2])
        else:
            bond_idx = get_bond_idx(bond_nbrs, nbr_list)

        return nbr_list, distance_feats, bond_idx

    def make_h(self,
               batch,
               r,
               xyz,
               offsets):
        """
        Initialize the hidden edge features.
        Args:
            batch (dict): batched sample of species
            r (torch.Tensor): initial atom features
            xyz (torch.Tensor): xyz of the batch
            offsets (float): periodic boundary condition offsets
        Returns:
            h_0 (torch.Tensor): initial hidden edge features
        """

        # get the directed bond list and bond features

        bond_nbrs, was_directed = make_directed(batch["bonded_nbr_list"])
        bond_feats = batch["bond_features"]
        device = bond_nbrs.device

        # if it wasn't directed before, repeat the bond features twice
        if not was_directed:
            bond_feats = torch.cat([bond_feats] * 2, dim=0)

        # get the distance-based edge features
        nbr_list, distance_feats, bond_idx = self.get_distance_feats(
            batch=batch,
            xyz=xyz,
            offsets=offsets,
            bond_nbrs=bond_nbrs)

        # combine node and bonded edge features to get the bond component
        # of h_0

        cp_bond_feats = self.W_i_cp(r=r,
                                    bond_feats=bond_feats,
                                    bond_nbrs=bond_nbrs)
        h_0_bond = torch.zeros((nbr_list.shape[0], cp_bond_feats.shape[1]))
        h_0_bond = h_0_bond.to(device)
        h_0_bond[bond_idx] = cp_bond_feats

        # combine node and distance edge features to get the schnet component
        # of h_0

        h_0_distance = self.W_i_schnet(r=r,
                                       bond_feats=distance_feats,
                                       bond_nbrs=nbr_list)

        # concatenate the two together

        h_0 = torch.cat([h_0_bond, h_0_distance], dim=-1)

        return h_0

    def convolve_sub_batch(self,
                           batch,
                           xyz=None,
                           xyz_grad=False):
        """
        Apply the convolution layers to a sub-batch.
        Args:
            batch (dict): batched sample of species
            xyz (torch.Tensor): xyz of the batch
            xyz_grad (bool): whether to set xyz.requires_grad = True
        Returns:
            new_node_feats (torch.Tensor): new node features after
                the convolutions.
            xyz (torch.Tensor): xyz of the batch
        """

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]

        if xyz_grad:
            xyz.requires_grad = True

        # get the directed neighbor list
        a, _ = make_directed(batch["nbr_list"])
        # get the atom features
        r = batch["atom_features"]
        # offsets for periodic boundary conditions
        offsets = batch.get("offsets", 0)
        # to deal with any shape mismatches
        if hasattr(offsets, 'max') and offsets.max() == 0:
            offsets = 0

        # initialize hidden bond features
        h_0 = self.make_h(batch=batch,
                          r=r,
                          xyz=xyz,
                          offsets=offsets)
        h_new = h_0.clone()

        # update edge features
        for conv in self.convolutions:
            h_new = conv(h_0=h_0,
                         h_new=h_new,
                         nbrs=a,
                         kj_idx=batch.get("kj_idx"),
                         ji_idx=batch.get("ji_idx"))

        # convert back to node features
        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a)

        return new_node_feats, xyz


class OnlyBondUpdateCP3D(ChemProp3D):

    def __init__(self, modelparams):
        """
        Initialize model.
        Args:
            modelparams (dict): dictionary of parameters for the model
        Returns:
            None
        """

        WeightedConformers.__init__(self, modelparams)

        input_layers = modelparams["input_layers"]
        output_layers = modelparams["output_layers"]

        # make the convolutions, the input network W_i, and the output
        # network W_o

        self.W_i = ChemPropInit(input_layers=input_layers)
        self.convolutions = self.make_convs(modelparams)
        self.W_o = ChemPropMsgToNode(
            output_layers=output_layers)

        # dimension of the hidden bond vector
        self.n_bond_hidden = modelparams["n_bond_hidden"]

    def make_convs(self, modelparams):
        """
        Make the convolution layers.
        Args:
            modelparams (dict): dictionary of parameters for the model
        Returns:
            convs (nn.ModuleList): list of networks for each convolution
        """

        num_conv = modelparams["n_convolutions"]
        same_filters = modelparams["same_filters"]

        # call `CpSchNetConv` to make the convolution layers
        convs = nn.ModuleList([CpSchNetConv(**modelparams)
                               for _ in range(num_conv)])

        # if you want to use the same filters for every convolution, repeat
        # the initial network and delete all the others
        if same_filters:
            convs = nn.ModuleList([convs[0] for _ in range(num_conv)])

        return convs

    def make_h(self,
               batch,
               nbr_list,
               r,
               nbr_was_directed):
        """
        Initialize the hidden bond features.
        Args:
            batch (dict): batched sample of species
            nbr_list (torch.LongTensor): neighbor list
            r (torch.Tensor): initial atom features
            nbr_was_directed (bool): whether the old neighbor list
                was directed or not
        Returns:
            h_0 (torch.Tensor): initial hidden bond features
            bond_nbrs (torch.LongTensor): bonded neighbor list
            bond_idx (torch.LongTensor): indices that map
                an element of `bond_nbrs` to the corresponding
                element in `nbr_list`. 
        """

        # get the directed bond list and bond features

        bond_nbrs, was_directed = make_directed(batch["bonded_nbr_list"])
        bond_feats = batch["bond_features"]
        device = bond_nbrs.device

        # if it wasn't directed before, repeat the bond features twice
        if not was_directed:
            bond_feats = torch.cat([bond_feats] * 2, dim=0)

        # initialize hidden bond features

        h_0_bond = self.W_i(r=r,
                            bond_feats=bond_feats,
                            bond_nbrs=bond_nbrs)

        # initialize `h_0`, the features of all edges
        # (including bonded ones), to zero

        nbr_dim = nbr_list.shape[0]
        h_0 = torch.zeros((nbr_dim,  self.n_bond_hidden))
        h_0 = h_0.to(device)

        # set the features of bonded edges equal to the bond
        # features

        if "bond_idx" in batch:
            bond_idx = batch["bond_idx"]
            if not nbr_was_directed:
                nbr_dim = nbr_list.shape[0]
                bond_idx = torch.cat([bond_idx,
                                      bond_idx + nbr_dim // 2])
        else:
            bond_idx = get_bond_idx(bond_nbrs, nbr_list)
            bond_idx = bond_idx.to(device)

        h_0[bond_idx] = h_0_bond

        return h_0, bond_nbrs, bond_idx

    def convolve_sub_batch(self,
                           batch,
                           xyz=None,
                           xyz_grad=False):
        """
        Apply the convolution layers to a sub-batch.
        Args:
            batch (dict): batched sample of species
            xyz (torch.Tensor): xyz of the batch
            xyz_grad (bool): whether to set xyz.requires_grad = True
        Returns:
            new_node_feats (torch.Tensor): new node features after
                the convolutions.
            xyz (torch.Tensor): xyz of the batch
        """

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]

        if xyz_grad:
            xyz.requires_grad = True

        a, nbr_was_directed = make_directed(batch["nbr_list"])
        # get the atom features
        r = batch["atom_features"]
        offsets = batch.get("offsets", 0)
        # to deal with any shape mismatches
        if hasattr(offsets, "max") and offsets.max() == 0:
            offsets = 0

        # get the distances between neighbors
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
             offsets).pow(2).sum(1).sqrt()[:, None]

        # initialize hidden bond features
        h_0, bond_nbrs, bond_idx = self.make_h(
            batch=batch,
            nbr_list=a,
            r=r,
            nbr_was_directed=nbr_was_directed)

        h_new = h_0.clone()

        # update edge features

        for conv in self.convolutions:

            # don't use any kj_idx or ji_idx
            # because they are only relevant when
            # you're doing updates with all neighbors, 
            # not with just the bonded neighbors like
            # we do here
            
            h_new = conv(h_0=h_0,
                         h_new=h_new,
                         all_nbrs=a,
                         bond_nbrs=bond_nbrs,
                         bond_idx=bond_idx,
                         e=e,
                         kj_idx=None,
                         ji_idx=None)

        # convert back to node features

        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a)

        return new_node_feats, xyz
