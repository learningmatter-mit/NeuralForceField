import torch
import torch.nn as nn


from nff.nn.layers import DEFAULT_DROPOUT_RATE
from nff.nn.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
)
from nff.nn.graphop import conf_pool
from nff.nn.utils import construct_sequential

import pdb

class WeightedConformers(nn.Module):

    def __init__(self, modelparams):
        """Constructs a SchNet model.

        Args:
            modelparams (TYPE): Description

        Example:

            n_atom_basis = 256
            mol_basis = 512

            mol_fp_layers = [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int((n_atom_basis + mol_basis)/2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int((n_atom_basis + mol_basis)/2),
                                                                          'out_features': mol_basis}}]

            readoutdict = {
                                "covid": [{'name': 'linear', 'param' : { 'in_features': mol_basis,
                                                                          'out_features': int(mol_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(mol_basis / 2),
                                                                          'out_features': 1}},
                                           {'name': 'sigmoid', 'param': {}}],
                            }


            modelparams = {
                'n_atom_basis': n_atom_basis,
                'n_filters': 256,
                'n_gaussians': 32,
                'n_convolutions': 4,
                'cutoff': 5.0,
                'trainable_gauss': True,
                'readoutdict': readoutdict,    
                'dropout_rate': 0.2
            }

            model = WeightedConformers(modelparams)

        """

        nn.Module.__init__(self)

        n_atom_basis = modelparams["n_atom_basis"]
        n_filters = modelparams["n_filters"]
        n_gaussians = modelparams["n_gaussians"]
        n_convolutions = modelparams["n_convolutions"]
        cutoff = modelparams["cutoff"]
        trainable_gauss = modelparams.get("trainable_gauss", False)
        dropout_rate = modelparams.get("dropout_rate", DEFAULT_DROPOUT_RATE)


        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        # convolutions
        self.convolutions = nn.ModuleList(
            [
                SchNetConv(
                    n_atom_basis=n_atom_basis,
                    n_filters=n_filters,
                    n_gaussians=n_gaussians,
                    cutoff=cutoff,
                    trainable_gauss=trainable_gauss,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )

        
        mol_fp_layers = modelparams["mol_fp_layers"]
        readoutdict = modelparams["readoutdict"]

        self.mol_fp_nn = construct_sequential(mol_fp_layers)
        self.readout = NodeMultiTaskReadOut(multitaskdict=readoutdict)

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
            xyz = batch["nxyz"][:, 1:4]
            xyz.requires_grad = True

        r = batch["nxyz"][:, 0]
        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        N = batch["num_atoms"].reshape(-1).tolist()
        num_confs = (torch.tensor(N) / torch.tensor(mol_sizes)).tolist()

        a = batch["nbr_list"]

        # offsets take care of periodic boundary conditions
        offsets = batch.get("offsets", 0)

        e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
             offsets).pow(2).sum(1).sqrt()[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        

        # split the fingerprints into fingerprints of the different conformers
        fps_by_smiles = torch.split(r, N)

        boltzmann_weights = torch.split(batch["weights"], num_confs)
        outputs = dict(r=r,
                       N=N,
                       xyz=xyz,
                       fps_by_smiles=fps_by_smiles,
                       boltzmann_weights=boltzmann_weights,
                       mol_sizes=mol_sizes)

        return outputs

    def forward(self, batch, xyz=None):
        """Summary

        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        outputs = self.convolve(batch, xyz)

        fps_by_smiles = outputs["fps_by_smiles"]
        batched_weights = outputs["boltzmann_weights"]
        mol_sizes = outputs["mol_sizes"]

        conf_fps = []
        for i in range(len(fps_by_smiles)):
            boltzmann_weights = batched_weights[i]
            smiles_fp = fps_by_smiles[i]
            mol_size = mol_sizes[i]

            conf_fp = conf_pool(smiles_fp=smiles_fp,
                                mol_size=mol_size,
                                boltzmann_weights=boltzmann_weights,
                                mol_fp_nn=self.mol_fp_nn)

            conf_fps.append(conf_fp)

        conf_fps = torch.stack(conf_fps)
        results = self.readout(conf_fps)

        print(results)

        return results
