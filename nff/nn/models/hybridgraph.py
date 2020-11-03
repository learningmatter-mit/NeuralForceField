import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.modules import SchNetConv, SchNetEdgeUpdate, NodeMultiTaskReadOut
from nff.nn.activations import shifted_softplus
from nff.nn.graphop import batch_and_sum
from nff.nn.utils import get_default_readout

from nff.utils.scatter import scatter_add

class HybridGraphConv(nn.Module):

    def __init__(self, modelparams):
        super().__init__()

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        trainable_gauss = modelparams.get('trainable_gauss', False)
        mol_n_convolutions = modelparams['mol_n_convolutions']
        mol_cutoff = modelparams['mol_cutoff']
        sys_n_convolutions = modelparams['sys_n_convolutions']
        sys_cutoff = modelparams['sys_cutoff']
        
        self.power = modelparams["V_ex_power"]
        self.sigma = torch.nn.Parameter(torch.Tensor([modelparams["V_ex_sigma"]]))

        # default predict var
        readoutdict = modelparams.get('readoutdict', get_default_readout(n_atom_basis))
        post_readout =  modelparams.get('post_readout', None)

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.molecule_convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff=mol_cutoff,
                             trainable_gauss=trainable_gauss,
                             dropout_rate=0.0)
            for _ in range(mol_n_convolutions)
        ])
        
        self.system_convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff=sys_cutoff,
                             trainable_gauss=trainable_gauss,
                             dropout_rate=0.0)
            for _ in range(sys_n_convolutions)
        ])

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(multitaskdict=readoutdict, post_readout=post_readout)        
        self.device = None
    
    def SeqConv(self, node, xyz, nbr_list, conv_module, pbc_offsets=None):
        if pbc_offsets is None:
            pbc_offsets = 0
        e = (xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]] + pbc_offsets).pow(2).sum(1).sqrt()[:, None]
        for i, conv in enumerate(conv_module):
            dr = conv(r=node, e=e, a=nbr_list)
            node = node + dr
        return node
    
    def V_ex(self, xyz, nbr_list, pbc_offsets):
        dist = (xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]] + pbc_offsets).pow(2).sum(1).sqrt()
        potential = ((dist.reciprocal() * self.sigma).pow(self.power))
        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]
    
    def forward(self, batch, **kwargs):
        r = batch['nxyz'][:, 0]
        xyz = batch['nxyz'][:, 1:4]
        N = batch['num_atoms'].reshape(-1).tolist()
        a_mol = batch['atoms_nbr_list']
        a_sys = batch['nbr_list']

        # offsets take care of periodic boundary conditions
        offsets = batch.get('offsets', 0) # offsets only affect nbr_list 
        xyz.requires_grad = True
        node_input = self.atom_embed(r.long()).squeeze()
        
        # system convolution 
        r_sys = self.SeqConv(node_input, xyz, a_sys, self.system_convolutions, offsets)
        r_mol = self.SeqConv(node_input, xyz, a_mol, self.molecule_convolutions)
        # Excluded Volume interactions 
        #r_ex = self.V_ex(xyz, a_sys, offsets)
        results = self.atomwisereadout(r_sys + r_mol)
        # add excluded volume interactions 
        #results['energy'] += r_ex
        results = batch_and_sum(results, N, list(batch.keys()), xyz)
        return results
