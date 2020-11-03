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

        # dimension of the hidden bond vector
        self.n_bond_hidden = modelparams["n_bond_hidden"]
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
        modelparams.update({"n_edge_hidden": modelparams["mol_basis"],
                            "dropout_rate": modelparams["schnet_dropout"]})

        # call `CpSchNetConv` to make the convolution layers
        convs = nn.ModuleList([ChemPropConv(**modelparams)
                               for _ in range(num_conv)])

        return convs

    def get_distance_feats(self,
                           batch,
                           xyz,
                           offsets):
        """
        Get distance features.
        Args:
            batch (dict): batched sample of species
            xyz (torch.Tensor): xyz of the batch
            offsets (float): periodic boundary condition offsets
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
            bonded_nbr_list = batch["bonded_nbr_list"]
            bond_idx = get_bond_idx(bonded_nbr_list, nbr_list)

        return nbr_list, distance_feats, bond_idx

    def make_h(self,
               batch,
               nbr_list,
               r,
               xyz,
               offsets):
        """
        Initialize the hidden edge features.
        Args:
            batch (dict): batched sample of species
            nbr_list (torch.LongTensor): neighbor list
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
            offsets=offsets)

        # combine node and bonded edge features to get the bond component
        # of h_0

        cp_bond_feats = self.W_i_cp(r=r,
                                    bond_feats=bond_feats,
                                    bond_nbrs=bond_nbrs)
        h_0_bond = torch.zeros((nbr_list.shape[0],  cp_bond_feats.shape[1]))
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

    def split_nbrs(self,
                   nbr_list,
                   mol_size,
                   num_confs,
                   confs_per_split):
        """
        Split neighbor list of a species into chunks for each sub-batch.
        Args:
            nbr_list (torch.LongTensor): neighbor list for
            mol_size (int): number of atoms in the molecule
            num_confs (int): number of conformers in the species
            confs_per_split (list[int]): number of conformers in each
                sub-batch.
        Returns:
            all_grouped_nbrs (list[torch.LongTensor]): list of 
                neighbor lists for each sub-batch.
        """

        # first split by conformer
        new_nbrs = []
        for i in range(num_confs):
            mask = (nbr_list[:, 0] <= (i + 1) * mol_size
                    ) * (nbr_list[:, 1] <= (i + 1) * mol_size)

            new_nbrs.append(nbr_list[mask])
            nbr_list = nbr_list[torch.bitwise_not(mask)]

        # regroup in sub-batches and subtract appropriately

        all_grouped_nbrs = []
        for i, num in enumerate(confs_per_split):

            prev_idx = sum(confs_per_split[:i])
            nbr_idx = list(range(prev_idx, prev_idx + num))

            grouped_nbrs = torch.cat([new_nbrs[i] for i in nbr_idx])
            grouped_nbrs -= mol_size * prev_idx

            all_grouped_nbrs.append(grouped_nbrs)

        return all_grouped_nbrs

    def fix_bond_idx(self, sub_batches):
        """
        Fix `bond_idx` when splitting batch into sub-batches.
        Args:
            sub_batches (list[dict]): sub-batches of the batch
        Returns:
            sub_batches (list[dict]): `sub_batches` with `bond_idx`
                fixed.
        """

        old_num_nbrs = 0
        for i, batch in enumerate(sub_batches):
            batch["bond_idx"] -= old_num_nbrs
            sub_batches[i] = batch

            old_num_nbrs += batch["nbr_list"].shape[0]
        return sub_batches

    def add_split_nbrs(self,
                       batch,
                       mol_size,
                       num_confs,
                       confs_per_split,
                       sub_batches):
        """
        Add split-up neighbor lists to each sub-batch.
        Args:
            batch (dict): batched sample of species
            mol_size (int): number of atoms in the molecule
            num_confs (int): number of conformers in the species
            confs_per_split (list[int]): number of conformers in each
                sub-batch.
            sub_batches (list[dict]): list of sub_batches
        Returns:
            sub_batches (list[dict]): list of sub_batches updated with
                their neighbor lists.
        """

        # go through each key that needs to be reindex as a neighbor list
        # (i.e. the neighbor list and the bonded neighbor list)

        for key in REINDEX_KEYS:
            if key not in batch:
                continue
            nbr_list = batch[key]
            split_nbrs = self.split_nbrs(nbr_list=nbr_list,
                                         mol_size=mol_size,
                                         num_confs=num_confs,
                                         confs_per_split=confs_per_split)
            for i, sub_batch in enumerate(sub_batches):
                sub_batch[key] = split_nbrs[i]
                sub_batches[i] = sub_batch
        return sub_batches

    def get_confs_per_split(self,
                            batch,
                            num_confs,
                            sub_batch_size):
        """
        Get the number of conformers per sub-batch.
        Args:
            batch (dict): batched sample of species
            num_confs (int): number of conformers in the species
            sub_batch_size (int): maximum number of conformers per
                sub-batch.
        Returns:
            confs_per_split (list[int]): number of conformers in each
                sub-batch.
        """

        val_len = len(batch["nxyz"])
        inherent_val_len = val_len // num_confs
        split_list = [sub_batch_size * inherent_val_len] * math.floor(
            num_confs / sub_batch_size)

        # if there's a remainder

        if sum(split_list) != val_len:
            split_list.append(val_len - sum(split_list))

        confs_per_split = [i // inherent_val_len for i in split_list]

        return confs_per_split

    def split_batch(self,
                    batch,
                    sub_batch_size):
        """
        Split a batch into sub-batches.
        Args:
            batch (dict): batched sample of species
            sub_batch_size (int): maximum number of conformers per
                sub-batch.
        Returns:
            sub_batches (list[dict]): sub batches of the batch
        """

        mol_size = batch["mol_size"].item()
        num_confs = len(batch["nxyz"]) // mol_size
        sub_batch_dic = {}

        confs_per_split = self.get_confs_per_split(
            batch=batch,
            num_confs=num_confs,
            sub_batch_size=sub_batch_size)

        num_splits = len(confs_per_split)

        for key, val in batch.items():
            val_len = len(val)
            if key in REINDEX_KEYS:
                continue
            elif np.mod(val_len, num_confs) != 0 or val_len == 1:
                if key == "num_atoms":
                    sub_batch_dic[key] = [int(val * num / num_confs)
                                          for num in confs_per_split]
                else:
                    sub_batch_dic[key] = [val] * num_splits
                continue

            # the per-conformer length of the value is `val_len`
            # divided by the number of conformers

            inherent_val_len = val_len // num_confs

            # use this to determine the number of items in each
            # section of the split list

            split_list = [inherent_val_len * num
                          for num in confs_per_split]

            # split the value accordingly
            split_val = torch.split(val, split_list)
            sub_batch_dic[key] = split_val

        sub_batches = [{key: sub_batch_dic[key][i] for key in
                        sub_batch_dic.keys()} for i in range(num_splits)]

        # fix neighbor list indexing
        sub_batches = self.add_split_nbrs(batch=batch,
                                          mol_size=mol_size,
                                          num_confs=num_confs,
                                          confs_per_split=confs_per_split,
                                          sub_batches=sub_batches)
        # fix `bond_idx`
        sub_batches = self.fix_bond_idx(sub_batches)

        return sub_batches

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

        # initialize hidden bond features
        h_0 = self.make_h(batch=batch,
                          nbr_list=a,
                          r=r,
                          xyz=xyz,
                          offsets=offsets)
        h_new = h_0.clone()

        # update edge features
        for conv in self.convolutions:
            h_new = conv(h_0=h_0,
                         h_new=h_new,
                         nbrs=a)

        # convert back to node features
        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a)

        return new_node_feats, xyz

    def convolve(self,
                 batch,
                 sub_batch_size=None,
                 xyz=None,
                 xyz_grad=False):
        """
        Apply the convolution layers to the batch.
        Args:
            batch (dict): batched sample of species
            sub_batch_size (int): maximum number of conformers
                in a sub-batch.
            xyz (torch.Tensor): xyz of the batch
            xyz_grad (bool): whether to set xyz.requires_grad = True
        Returns:
            new_node_feats (torch.Tensor): new node features after
                the convolutions.
            xyz (torch.Tensor): xyz of the batch
        """

        # split batches as necessary
        if sub_batch_size is None:
            sub_batches = [batch]
        else:
            sub_batches = self.split_batch(batch, sub_batch_size)

        # go through each sub-batch, get the xyz and node features,
        # and concatenate them when done

        new_node_feat_list = []
        xyz_list = []

        for sub_batch in sub_batches:

            new_node_feats, xyz = self.convolve_sub_batch(
                sub_batch, xyz, xyz_grad)
            new_node_feat_list.append(new_node_feats)
            xyz_list.append(xyz)

        new_node_feats = torch.cat(new_node_feat_list)
        xyz = torch.cat(xyz_list)

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
            bonded_nbr_list = batch["bonded_nbr_list"]
            bond_idx = get_bond_idx(bonded_nbr_list, nbr_list)

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

            h_new = conv(h_0=h_0,
                         h_new=h_new,
                         all_nbrs=a,
                         bond_nbrs=bond_nbrs,
                         bond_idx=bond_idx,
                         e=e)

        # convert back to node features

        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a)

        return new_node_feats, xyz
