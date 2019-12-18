import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.modules import (SchNetConv, BondEnergyModule, SchNetEdgeUpdate, NodeMultiTaskReadOut,
                            AuTopologyReadOut, DoubleNodeConv, SingleNodeConv)
from nff.nn.activations import shifted_softplus
from nff.nn.graphop import batch_and_sum, get_atoms_inside_cell
from nff.nn.utils import get_default_readout
from nff.utils.scatter import compute_grad
import numpy as np
import pdb


STRING_TO_MODULE = {
    "double_node": DoubleNodeConv,
    "single_node": SingleNodeConv
}


class SchNet(nn.Module):

    """SchNet implementation with continous filter.

    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis
        convolutions (torch.nn.Module): convolution layers applied to the graph
        atomwisereadout (torch.nn.Module): fully connected layers applied to the graph
            to get the results of interest
        device (int): GPU being used.


    """

    def __init__(self, modelparams):
        """Constructs a SchNet model.

        Args:
            modelparams (TYPE): Description

        Example:

            n_atom_basis = 256

            readoutdict = {
                                "energy_0": [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int(n_atom_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2),
                                                                          'out_features': 1}}],
                                "energy_1": [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int(n_atom_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2),
                                                                          'out_features': 1}}]
                            }


            modelparams = {
                'n_atom_basis': n_atom_basis,
                'n_filters': 256,
                'n_gaussians': 32,
                'n_convolutions': 4,
                'cutoff': 5.0,
                'trainable_gauss': True,
                'readoutdict': readoutdict,    


            }

            model = SchNet(modelparams)

        """

        nn.Module.__init__(self)

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        n_convolutions = modelparams['n_convolutions']
        cutoff = modelparams['cutoff']
        trainable_gauss = modelparams.get('trainable_gauss', False)


        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        readoutdict = modelparams.get(
            'readoutdict', get_default_readout(n_atom_basis))
        post_readout = modelparams.get('post_readout', None)


        # convolutions
        self.convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                       n_filters=n_filters,
                       n_gaussians=n_gaussians,
                       cutoff=cutoff,
                       trainable_gauss=trainable_gauss)
            for _ in range(n_convolutions)
        ])

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(
            multitaskdict=readoutdict, post_readout=post_readout)
        self.device = None




    def convolve(self, batch, xyz=None):
        """

        Apply the convolutional layers to the batch.

        Args:
            batch (dict): dictionary of props

        Returns:
            r: new feature vector after the convolutions
            N: list of the number of atoms for each molecule in the batch
            xyz: xyz (with a "requires_grad") for the batch
        """

        # Note: we've given the option to input xyz from another source.
        # E.g. if you already created an xyz  and set requires_grad=True,
        # you don't want to make a whole new one.

        if xyz is None:
            xyz = batch['nxyz'][:, 1:4]
            xyz.requires_grad = True

        r = batch['nxyz'][:, 0]
        N = batch['num_atoms'].reshape(-1).tolist()
        a = batch['nbr_list']

        # offsets take care of periodic boundary conditions
        offsets = batch.get('offsets', 0)
        
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] - offsets).pow(2).sum(1).sqrt()[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        return r, N, xyz

    def forward(self, batch, xyz=None):
        """Summary

        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        r, N, xyz = self.convolve(batch, xyz)
        r = self.atomwisereadout(r)
        results = batch_and_sum(r, N, list(batch.keys()), xyz)

        return results


class AuTopology(nn.Module):

    """
    AuTopology model for getting classical forces.
    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis
        convolutions (torch.nn.Module): convolution layers applied to the graph
        atomwisereadout (torch.nn.Module): fully connected layers applied to the graph
            to get the results of interest
        device (int): GPU being used.

    """

    def __init__(self, modelparams):

        """
        Constructs an AuTopology model.
        Example:
            n_autopology_features = 256

            modelparams = {
                "n_features": n_autopology_features,
                "n_convolutions": 4,
                "conv_type": "double_node",


                "conv_update_layers": [{'name': 'linear', 'param' : {'in_features': int(2*n_autopology_features),
                                                                'out_features': n_autopology_features}},
                                  {'name': 'Tanh', 'param': {}},
                                  {'name': 'linear', 'param' : {'in_features': n_autopology_features,
                                                          'out_features': n_autopology_features}},
                                  {'name': 'Tanh', 'param': {}}

                        ],
                
                
                "readout_hidden_nodes": [40, 20],
                
                "bond_terms": ["morse"],
                "angle_terms": ["harmonic"],
                "dihedral_terms": ["OPLS"],
                "improper_terms": ["harmonic"],
                "pair_terms": ["LJ"],
                
                "output_keys": ["energy_0", "energy_1"],
                "trainable_prior": True

            }

            model = AuTopology(modelparams)


        """

        nn.Module.__init__(self)

        n_features = modelparams["n_features"]
        update_layers = modelparams["conv_update_layers"]
        conv_type = modelparams["conv_type"]
        n_convolutions = modelparams["n_convolutions"]

        self.atom_embed = nn.Embedding(100, n_features, padding_idx=0)

        self.convolutions = nn.ModuleList([
            STRING_TO_MODULE[conv_type](update_layers)
            for _ in range(n_convolutions)
        ])


        Lh = modelparams["readout_hidden_nodes"]
        Fr = modelparams["n_features"]
        modelparams.update({"Lh": Lh, "Fr": Fr})

        self.readout = AuTopologyReadOut(multitaskdict=modelparams)
        self.device = None


    def convolve(self, batch):

        """

        Apply the convolutional layers to the batch.

        Args:
            batch (dict): dictionary of props

        Returns:
            r: new feature vector after the convolutions
        """

        # not implemented for PBC yet (?)

        a = batch["bonded_nbr_list"]
        z = batch['nxyz'][:, 0]
        r = self.atom_embed(z.long()).squeeze()

        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, a=a, e=None)
            r = r + dr

        return r

    def forward(self, batch, xyz=None):

        """Summary

        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        # Give the option to input xyz from another source. E.g. if you already created
        # an xyz with schnet and set requires_grad=True, you don't want to make a whole
        # new one.

        if xyz is None:
            xyz = batch['nxyz'][:, 1:4]
            xyz.requires_grad = True

        r = self.convolve(batch)
        results = self.readout(r=r, batch=batch, xyz=xyz)
        return results


class SchNetAuTopology(nn.Module):

    """
    A neural network model that combines AuTopology with SchNet.
    Attributes:


        schnet (models.SchNet): SchNet model
        autopology (models.AuTopology): AuTopology model

        sorted_result_keys (list): names of energies to output, sorted from lowest to highest
            (e.g. ["energy_0", "energy_1", "energy_2"]) 
        grad_keys (list): same as sorted_result_keys, but with "_grad" at the end for gradients
        sort_results (bool): whether or not to sort results so that the energies are guaranteed
            to be ordered properly

        sadd_autopology (bool): use the autopology module in the final results
        add_schnet (bool): use the schnet module in the final results



    """

    def __init__(self, modelparams, add_autopology=True, add_schnet=False):
        """Constructs a SchNet model.

        Args:
            modelparams (dict): dictionary of parameters for the model
            add_autpology (bool): add the autopology result to the final energy
            add_schnet (bool): add the schnet result to the final energy

        Returns:
            None

        Example:


            # given `autopology_params` and `schnet_params`, like the ones given as examples in the SchNet and
            # AuTopology docstrings:

            modelparams = {
                
                "autopology_params": autopology_params,
                "schnet_params": schnet_params,
                "sorted_result_keys": ["energy_0", "energy_1"],
                "grad_keys": ["energy_0_grad", "energy_1_grad"],
                "sort_results": False,
                
            }

            example_module = SchNetAuTopology(modelparams, add_autopology=True, add_schnet=False)

        """

        super().__init__()

        schnet_params = modelparams["schnet_params"]
        self.schnet = SchNet(schnet_params)

        autopology_params = modelparams["autopology_params"]
        self.autopology = AuTopology(autopology_params)


        # Add some other useful attributes
        self.sorted_result_keys = modelparams["sorted_result_keys"]
        self.grad_keys = modelparams["grad_keys"]
        self.sort_results = modelparams["sort_results"]


        # Decide whether to add the autopology and/or schnet results to the final answer
        self.add_autopology = add_autopology
        self.add_schnet = add_schnet


    def transfer_to_schnet(self):
        """
        Shift the learning from AuTopology to SchNet.
        """

        # Freeze the AuTopology parameters so they can no longer be learned.

        for param in self.autopology.parameters():
            param.requires_grad = False

        # Start adding the SchNet result to the final result
        self.add_schnet = True


    def get_sorted_results(self, pre_results, num_atoms):

        """
        Sort results by energies.
        Args:
            pre_results (dict): dictionary of the network output before sorting
            num_atoms (torch.tensor): array of number of atoms per molecule
        Returns:
            final_results (dict): dictionary of the network output after sorting
        """

        # sort the energies for each molecule in the batch and put the results in
        # `final_results`.
        batch_length = len(pre_results[self.sorted_result_keys[0]])
        final_results = {key: [] for key in [*self.sorted_result_keys, *self.grad_keys]}

        # de-batch the gradients
        N = num_atoms.cpu().numpy().tolist()
        for key in self.grad_keys:
            pre_results[key] = torch.split(pre_results[key], N)


        for i in range(batch_length):
            # sort the outputs
            sorted_energies, sorted_idx = torch.sort(torch.cat([pre_results[key][i] for key in
                                                                self.sorted_result_keys]))

            for index, energy_key in zip(sorted_idx, self.sorted_result_keys):

                corresponding_key = self.sorted_result_keys[index]

                final_results[energy_key].append(pre_results[corresponding_key][i])
                final_results[energy_key + "_grad"].append(pre_results[corresponding_key + "_grad"][i])


        # re-batch the outputs  
        for key in self.sorted_result_keys:
            final_results[key] = torch.stack(final_results[key])
            final_results[key + "_grad"] = torch.cat(final_results[key + "_grad"])

        return final_results


    def forward(self, batch):
        """
        Applies the neural network to a batch.
        Args:
            batch (dict): dictionary of props
        Returns:
            final_results (dict): A dictionary of results for each key in
                self.sorted_results_keys and self.grad_keys. 

        """


        # define xyz here to avoid making two graphs (one for schnet and
        # one for autopology)

        xyz = batch["nxyz"][:, 1:4]
        xyz.requires_grad = True

        if self.add_schnet:
            schnet_results = self.schnet(batch=batch, xyz=xyz)

        if self.add_autopology:
            auto_results = self.autopology(batch=batch, xyz=xyz)

        # pre_results is the dictionary of results before sorting energies
        pre_results = dict()

        for key in [*self.sorted_result_keys, *self.grad_keys]:
            if self.add_schnet:
                pre_results[key] = schnet_results[key]

            if self.add_schnet and self.add_autopology:
                pre_results[key] += auto_results[key]

            elif self.add_autopology and not self.add_schnet:
                pre_results[key] = auto_results[key]


        # sort the results if necessary
        if self.sort_results:
            final_results = self.get_sorted_results(pre_results, batch["num_atoms"])
        else:
            final_results = pre_results


        return final_results



