from torch import nn
import torch
import numpy as np
import math

from nff.nn.models.conformers import WeightedConformers
from nff.nn.modules import (CpSchNetConv, ChemPropMsgToNode,
                            ChemPropInit)
from nff.utils.tools import make_directed
from nff.data.loader import REINDEX_KEYS


class SchNetFeatures(WeightedConformers):

    def __init__(self, modelparams):

        WeightedConformers.__init__(self, modelparams)
        delattr(self, "atom_embed")

        input_layers = modelparams["input_layers"]
        output_layers = modelparams["output_layers"]

        self.W_i = ChemPropInit(input_layers=input_layers)
        self.convolutions = self.make_convs(modelparams)
        self.W_o = ChemPropMsgToNode(
            output_layers=output_layers)

        self.n_bond_hidden = modelparams["n_bond_hidden"]

    def make_convs(self, modelparams):

        num_conv = modelparams["n_convolutions"]
        same_filters = modelparams["same_filters"]

        convs = nn.ModuleList([CpSchNetConv(**modelparams)
                               for _ in range(num_conv)])
        if same_filters:
            convs = nn.ModuleList([convs[0] for _ in range(num_conv)])

        return convs

    def make_h(self, batch, nbr_list, r):
        """
        Initialize the hidden bond features
        """

        # get the directed bond list and bond features

        bond_nbrs, was_directed = make_directed(batch["bonded_nbr_list"])
        bond_feats = batch["bond_features"]
        device = bond_nbrs.device

        if not was_directed:
            bond_feats = torch.cat([bond_feats] * 2, dim=0)

        # initialize hidden bond features

        # Need to change this if we're going to only use
        # bonded_nbr_list for a single conformer and then applying them
        # to everything.

        h_0_bond = self.W_i(r=r,
                            bond_feats=bond_feats,
                            bond_nbrs=bond_nbrs)

        # initialize `h_0`, the features of all edges
        # (including non-bonded ones), to zero

        nbr_dim = nbr_list.shape[0]
        h_0 = torch.zeros((nbr_dim,  self.n_bond_hidden))
        h_0 = h_0.to(device)

        # set the features of bonded edges equal to the bond
        # features

        bond_idx = (bond_nbrs[:, None] == nbr_list
                    ).prod(-1).nonzero()[:, 1]
        h_0[bond_idx] = h_0_bond

        return h_0, bond_nbrs, bond_idx

    def split_nbrs(self, nbr_list, mol_size, num_confs, confs_per_split):

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

    def add_split_nbrs(self,
                       batch,
                       mol_size,
                       num_confs,
                       confs_per_split,
                       sub_batches):

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

    def get_confs_per_split(self, batch, num_confs, sub_batch_size):

        val_len = len(batch["nxyz"])
        inherent_val_len = val_len // num_confs
        split_list = [sub_batch_size * inherent_val_len] * math.floor(
            num_confs / sub_batch_size)

        # if there's a remainder

        if sum(split_list) != val_len:
            split_list.append(val_len - sum(split_list))

        confs_per_split = [i // inherent_val_len for i in split_list]

        return confs_per_split

    def split_batch(self, batch, sub_batch_size):
        """
        Only applies when the batch size is 1 (?)
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
            elif np.mod(val_len, num_confs) != 0:
                sub_batch_dic[key] = [val] * num_splits
                continue

            inherent_val_len = val_len // num_confs
            split_list = [inherent_val_len * num
                          for num in confs_per_split]

            split_val = torch.split(val, split_list)
            sub_batch_dic[key] = split_val

        sub_batches = [{key: sub_batch_dic[key][i] for key in
                        sub_batch_dic.keys()} for i in range(num_splits)]

        sub_batches = self.add_split_nbrs(batch=batch,
                                          mol_size=mol_size,
                                          num_confs=num_confs,
                                          confs_per_split=confs_per_split,
                                          sub_batches=sub_batches)

        return sub_batches

    def convolve_sub_batch(self, batch, xyz=None, xyz_grad=False):

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]

        if xyz_grad:
            xyz.requires_grad = True

        a, _ = make_directed(batch["nbr_list"])
        # a, _ = make_directed(batch["bonded_nbr_list"])

        r = batch["atom_features"]
        offsets = batch.get("offsets", 0)
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
             offsets).pow(2).sum(1).sqrt()[:, None]

        # initialize hidden bond features

        h_0, bond_nbrs, bond_idx = self.make_h(batch=batch,
                                               nbr_list=a,
                                               r=r)

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

    def convolve(self, batch, sub_batch_size=5, xyz=None, xyz_grad=False):

        sub_batches = self.split_batch(batch, sub_batch_size)
        new_node_feat_list = []
        xyz_list = []

        for sub_batch in sub_batches:
            new_node_feats, xyz = self.convolve_sub_batch(batch, xyz, xyz_grad)
            new_node_feat_list.append(new_node_feats)
            xyz_list.append(xyz)

        new_node_feats = torch.cat(new_node_feat_list)
        xyz = torch.cat(xyz_list)

        return new_node_feats, xyz
