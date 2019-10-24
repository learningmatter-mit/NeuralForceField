import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.modules import (GraphDis, SchNetConv, BondEnergyModule, SchNetEdgeUpdate, NodeMultiTaskReadOut,
                            AuTopologyReadOut)
from nff.nn.activations import shifted_softplus
from nff.nn.graphop import batch_and_sum, get_atoms_inside_cell, diagonalize, batch_energies
from nff.nn.utils import get_default_readout
from nff.utils.scatter import compute_grad
import numpy as np
import pdb


class SchNet(nn.Module):

    """SchNet implementation with continous filter.
    
    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis

        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        prop_dics (dict): A dictionary of the form {name: prop_dic}, where name is the
            property name and prop_dic is a dictionary for that property.
        module_dict (ModuleDict): a dictionary of modules. Each entry has the form
            {name: mod_list}, where name is the name of a property object and mod_list
            is a ModuleList of layers to predict that property.
    """
    
    def __init__(self, modelparams):
        """Constructs a SchNet model.
        
        Args:
            modelparams (TYPE): Description
        """

        super().__init__()

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        n_convolutions = modelparams['n_convolutions']
        cutoff = modelparams['cutoff']
        trainable_gauss = modelparams.get('trainable_gauss', False)

        # default predict var
        readoutdict = modelparams.get('readoutdict', get_default_readout(n_atom_basis))
        post_readout =  modelparams.get('post_readout', None)

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff=cutoff,
                             trainable_gauss=trainable_gauss)
            for _ in range(n_convolutions)
        ])

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(multitaskdict=readoutdict, post_readout=post_readout)        
        self.device = None

    def forward(self, batch, other_results=False):

        """Summary
        
        Args:
            batch (dict): dictionary of props
        
        Returns:
            dict: dionary of results 
        """
        r = batch['nxyz'][:, 0]
        xyz = batch['nxyz'][:, 1:4]
        N = batch['num_atoms'].reshape(-1).tolist()
        a = batch['nbr_list']

        # offsets take care of periodic boundary conditions
        offsets = batch.get('offsets', 0)

        xyz.requires_grad = True

        # calculating the distances
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] - offsets).pow(2).sum(1).sqrt()[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        r = self.atomwisereadout(r)
        results = batch_and_sum(r, N, list(batch.keys()), xyz)
        
        return results 




class SchNetAuTopology(nn.Module):

    """

    """


    def __init__(self, modelparams):
        """Constructs a SchNet model.
        
        Args:
            modelparams (dict): { 
                                  "schnet_keys": ["d0", "d1", "lambda"],
                                  "autopology_keys": ["d0_auto", "d1_auto"],
                                  "key_pairs": [{"autopology": "d0_auto", "schnet": "d0"},
                                                {"autopology": "d1_auto", "schnet": "d1"}],
                                  "matrix_keys": [["d0", "lambda"], ["lambda", "d1"]]
                                  "sorted_result_keys": ["energy_0", "energy_1"]},
                                  "grad_keys": ["energy_0_grad", "energy_1_grad"],


                                  "schnet_readout": [ ... ],

                                  "autopology_Lh": [40, 20], 
                                  "bond_terms": ["morse"],
                                  "angle_terms": ["harmonic"], 
                                  "dihedral_terms": ["OPLS"], 
                                  "improper_terms": ["harmonic"],
                                  "pair_terms": ["LJ"],


                                  }


        """

        super().__init__()

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        n_convolutions = modelparams['n_convolutions']
        cutoff = modelparams['cutoff']
        trainable_gauss = modelparams.get('trainable_gauss', False)

        self.sorted_result_keys = modelparams["sorted_result_keys"]
        self.schnet_keys = modelparams["schnet_keys"]
        self.autopology_keys = modelparams["autopology_keys"]
        self.key_pairs = modelparams["key_pairs"]
        self.matrix_keys = modelparams["matrix_keys"]
        self.grad_keys = modelparams["grad_keys"]

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff=cutoff,
                             trainable_gauss=trainable_gauss)
            for _ in range(n_convolutions)
        ])

        # ReadOut
        schnet_readout = modelparams.get('schnet_readout', get_default_readout(n_atom_basis))
        auto_readout = copy.deepcopy(modelparams)
        auto_readout.update({"Fr": n_atom_basis})
        auto_readout.update({"Lh": modelparams["autopology_Lh"]})

        self.schnet_readout = NodeMultiTaskReadOut(multitaskdict=schnet_readout) 
        self.auto_readout = AuTopologyReadOut(multitaskdict=auto_readout)

        self.device = None

    def convolve(self, batch):

        """Summary
        
        Args:
            batch (dict): dictionary of props
        
        Returns:
            dict: dionary of results 
        """

        # pdb.set_trace()

        r = batch['nxyz'][:, 0]
        xyz = batch['nxyz'][:, 1:4]
        N = batch['num_atoms'].reshape(-1).tolist()
        a = batch['nbr_list']

        # offsets take care of periodic boundary conditions
        offsets = batch.get('offsets', 0)

        xyz.requires_grad = True

        # calculating the distances
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] - offsets).pow(2).sum(1).sqrt()[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        return r, N, xyz

    def get_diabatic_results(self, batch):

        r, N, xyz = self.convolve(batch)
        schnet_r = self.schnet_readout(r)
        schnet_results = batch_energies(schnet_r, N, self.schnet_keys, xyz)
        auto_results = self.auto_readout(r=r, batch=batch, xyz=xyz)

        diabatic_results = dict()
        schnet_keys_in_pairs = []

        for key_pair in self.key_pairs:

            schnet_key = key_pair["schnet"]
            auto_key = key_pair["autopology"]
            schnet_keys_in_pairs.append(schnet_key)

            diabatic_results[schnet_key] = schnet_results[schnet_key] + auto_results[auto_key]
            # diabatic_results[schnet_key] = schnet_results[schnet_key]
        
        for key in self.schnet_keys:
            if key not in schnet_keys_in_pairs:
                diabatic_results[key] = schnet_results[schnet_key]

        return diabatic_results, xyz

    def forward(self, batch):

        diabatic_results, xyz = self.get_diabatic_results(batch)
        final_results = {key: [] for key in self.sorted_result_keys}
        batch_length = len(diabatic_results[self.matrix_keys[0][0]])

        # pdb.set_trace()

        # get energies by diagonalizing diabatic Hamiltonian
        for i in range(batch_length):
            matrix = []
            for key_row in self.matrix_keys:
                matrix.append([diabatic_results[key][i] for key in key_row])
            matrix = np.array(matrix)
            eigvals = diagonalize(matrix)
            # eigvals = [matrix[0, 0], matrix[1, 1]]

            for j, key in enumerate(self.sorted_result_keys):
                final_results[key].append(eigvals[j])

        # compute gradients and add them to final_results
        for key in self.sorted_result_keys:
            final_results[key] = torch.cat(final_results[key])

            if "{}_grad".format(key) not in self.grad_keys:
                continue    

            grad = compute_grad(inputs=xyz, output=final_results[key])
            final_results[key + "_grad"] = grad

        return final_results 

