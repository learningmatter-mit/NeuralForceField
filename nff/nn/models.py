import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.modules import (GraphDis, SchNetConv, BondEnergyModule, SchNetEdgeUpdate, NodeMultiTaskReadOut,
                            AuTopologyReadOut)
from nff.nn.activations import shifted_softplus
from nff.nn.graphop import batch_and_sum, get_atoms_inside_cell, batch_energies
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
    A neural network model that combines AuTopology with SchNet.
    Attributes:

    
        sorted_result_keys (list): a list of energies that you want the network to predict.
            These keys should be ordered by energy (e.g. ["energy_0", "energy_1"]).
        grad_keys (list): A list of gradients that you want the network to give (all members
            of this list should be elements of sorted_result_keys with "_grad" at the end)
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis
        convolutions (torch.nn.ModuleList): include all the convolutions
        schnet_readout (nn.Module): a module for reading out results from SchNet
        auto_readout (nn.Module): a module for reading out results from AuTopology
        device (int): GPU device number

    """


    def __init__(self, modelparams):
        """Constructs a SchNet model.
        
        Args:
            modelparams (dict): dictionary of parameters for the model
        Returns:
            None

        Example:

            modelparams =  { 
                              "sorted_result_keys": ["energy_0", "energy_1"],
                              "grad_keys": ["energy_0_grad", "energy_1_grad"],

                              "n_atom_basis": 256,
                              "n_filters": 256,
                              "n_gaussians": 32,
                              "n_convolutions": 4,
                              "cutoff": 5.0,
                              "trainable_gauss": True,

                              "schnet_readout": {"energy_0":
                                        [
                                            {'name': 'Dense', 'param': {'in_features': 5, 'out_features': 20}},
                                            {'name': 'shifted_softplus', 'param': {}},
                                            {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                                        ],

                                    "energy_1":
                                        [
                                            {'name': 'linear', 'param': {'in_features': 5, 'out_features': 20}},
                                            {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                                        ]
                                }, # parameters for the SchNet part of the readout
    

                              "trainable_prior": True, # whether the AuTopology parameters are learnable or not
                              "autopology_Lh": [40, 20], # layer parameters for AuTopology
                              "bond_terms": ["morse"], # type of classical bond prior
                              "angle_terms": ["harmonic"], # type of classical angle prior
                              "dihedral_terms": ["OPLS"],  # type of classical dihedral prior
                              "improper_terms": ["harmonic"], # type of classical improper prior
                              "pair_terms": ["LJ"], # type of classical non-bonded pair prior

                            }

                example_module = SchNetAuTopology(modelparams)

        """

        super().__init__()

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        n_convolutions = modelparams['n_convolutions']
        cutoff = modelparams['cutoff']
        trainable_gauss = modelparams.get('trainable_gauss', False)

        self.sorted_result_keys = modelparams["sorted_result_keys"]
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
        auto_readout.update({"Fr": n_atom_basis, "Lh": modelparams.get("autopology_Lh")})

        self.schnet_readout = NodeMultiTaskReadOut(multitaskdict=schnet_readout) 
        self.auto_readout = AuTopologyReadOut(multitaskdict=auto_readout)

        self.device = None

    def convolve(self, batch):

        """

        Apply the convolutional layers to the batch.
        
        Args:
            batch (dict): dictionary of props
        
        Returns:
            r: new feature vector after the convolutions
            N: list of the number of atoms for each molecule in the batch
            xyz: xyz (with a "requires_grad") for the batch
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

        return r, N, xyz

    def forward(self, batch):

        """
        Applies the neural network to a batch.
        Args:
            batch (dict): dictionary of props
        Returns:
            final_results (dict): A dictionary of results for each key in
                self.sorted_results_keys and self.grad_keys. Also contains
                results of just the autopology part of the calculation, in
                case you want to also minimize the force error with respect
                to the autopology calculation.

        """

        # get features, N, and xyz from the convolutions
        r, N, xyz = self.convolve(batch)
        # apply the SchNet readout to r
        schnet_r = self.schnet_readout(r)
        # get the SchNet results for the energies by batching them
        schnet_results = batch_energies(schnet_r, N, self.sorted_result_keys, xyz)
        # get the autopology results, which are automatically batched
        auto_results = self.auto_readout(r=r, batch=batch, xyz=xyz)

        # pre_results is the dictionary of results before sorting energies,
        # and autopology_results is the set of results only from autopology
        pre_results = dict()
        autopology_results = dict()

        for key in self.sorted_result_keys:

            # get the autopology results and their gradients
            autopology_results[key] = auto_results[key]
            autopology_grad = compute_grad(inputs=xyz, output=autopology_results[key])
            autopology_results[key + "_grad"] = autopology_grad

            # get pre_results by adding schnet_results to auto_results
            pre_results[key] = schnet_results[key] + auto_results[key]

        # sort the energies for each molecule in the batch and put the results in
        # `final_results`.
        batch_length = len(pre_results[self.sorted_result_keys[0]])
        final_results = {key: [] for key in self.sorted_result_keys}

        for i in range(batch_length):
            # pdb.set_trace()
            # sort the outputs and take the zeroth element (the zeroth element is the sorted
            # result, and the first element is the indices)
            sorted_energies = torch.sort(torch.cat([pre_results[key][i] for key in
                self.sorted_result_keys]))[0]
            for key, sorted_energy in zip(self.sorted_result_keys, sorted_energies):
                final_results[key].append(sorted_energy)

        for key in self.sorted_result_keys:
            final_results[key] = torch.stack(final_results[key])

            if "{}_grad".format(key) not in self.grad_keys:
                continue

            # compute the gradient with respect to the sorted energies
            grad = compute_grad(inputs=xyz, output=final_results[key])
            final_results[key + "_grad"] = grad

            # add the autopology results by putting "auto_" in front of each key

            final_results["auto_{}".format(key)] = autopology_results[key]
            final_results["auto_{}_grad".format(key)] = autopology_results["{}_grad".format(key)]

        return final_results


