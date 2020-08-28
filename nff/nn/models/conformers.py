import torch
import torch.nn as nn
import sys

from nff.nn.layers import DEFAULT_DROPOUT_RATE
from nff.nn.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
    ConfAttention
)
from nff.nn.graphop import conf_pool
from nff.nn.utils import construct_sequential
from nff.utils.scatter import compute_grad

"""
Model that uses a representation of a molecule in terms of different 3D
conformers to predict properties.
"""

# FEAT_SCALING = 20
FEAT_SCALING = 1


class WeightedConformers(nn.Module):

    def __init__(self, modelparams):
        """Constructs a SchNet-Like model using a conformer representation.

        Args:
            modelparams (dict): dictionary of parameters for model. All
                are the same as in SchNet, except for  `mol_fp_layers`,
                which describes how to convert atomic fingerprints into
                a single molecular fingerprint.

        Example:

            n_atom_basis = 256
            mol_basis = 512

            # all the atomic fingerprints get added together, then go through the network created
            # by `mol_fp_layers` to turn into a molecular fingerprint
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

            # dictionary to tell you what to do with the Boltzmann factors
            # ex. 1:

            boltzmann_dict = {"type": "multiply"}

            # ex. 2
            boltzmann_layers = [{'name': 'linear', 'param': {'in_features': mol_basis + 1,
                                                           'out_features': mol_basis}},
                                {'name': 'shifted_softplus', 'param': {}},
                                {'name': 'linear', 'param': {'in_features': mol_basis,
                                                           'out_features': mol_basis}}]
            boltzmann_dict = {"type": "layers", "layers": boltzmann_layers}


            modelparams = {
                'n_atom_basis': n_atom_basis,
                'n_filters': 256,
                'n_gaussians': 32,
                'n_convolutions': 4,
                'cutoff': 5.0,
                'trainable_gauss': True,
                'readoutdict': readoutdict,
                'mol_fp_layers': mol_fp_layers,
                'boltzmann_dict': boltzmann_dict
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

        # extra features to consider
        self.extra_feats = self.make_extra_feats(modelparams)

        mol_fp_layers = modelparams["mol_fp_layers"]
        readoutdict = modelparams["readoutdict"]
        boltzmann_dict = modelparams["boltzmann_dict"]

        # the nn that converts atomic finerprints to a molecular fp
        self.mol_fp_nn = construct_sequential(mol_fp_layers)

        # create a module that lets a molecular fp interact with the
        # conformer's boltzmann weight to give a final molecular fp
        self.boltz_nns = self.make_boltz_nn(boltzmann_dict)
        self.head_pool = boltzmann_dict.get("head_pool", "concatenate")

        # the readout acts on this final molceular fp
        self.readout = NodeMultiTaskReadOut(multitaskdict=readoutdict)

        # whether to learn the embeddings or get them from the batch
        self.batch_embeddings = modelparams.get("batch_embeddings", False)

    def make_extra_feats(self, modelparams):
        """
        Example:
            "extra_features": [{"name": "morgan", "length": 1048},
                              {"name": "rdkit_2d", "length": 120}]
        Returns:
            ["morgan", "rdkit_2d"]
        """
        if modelparams.get("extra_features") is None:
            return
        feat_dics = modelparams["extra_features"]
        feat_names = [dic["name"] for dic in feat_dics]
        return feat_names

    def make_boltz_nn(self, boltzmann_dict):

        networks = nn.ModuleList([])

        if boltzmann_dict["type"] == "multiply":
            return

        elif boltzmann_dict["type"] == "layers":
            layers = boltzmann_dict["layers"]
            networks.append(construct_sequential(layers))

        elif boltzmann_dict["type"] == "attention":

            num_heads = boltzmann_dict.get("num_heads", 1)
            equal_weights = boltzmann_dict.get("equal_weights", False)

            for _ in range(num_heads):

                mol_basis = boltzmann_dict["mol_basis"]
                boltz_basis = boltzmann_dict["boltz_basis"]
                final_act = boltzmann_dict["final_act"]

                networks.append(ConfAttention(mol_basis=mol_basis,
                                              boltz_basis=boltz_basis,
                                              final_act=final_act,
                                              equal_weights=equal_weights))

        return networks

    def add_features(self, batch, **kwargs):

        N = batch["num_atoms"].reshape(-1).tolist()
        num_mols = len(N)

        if self.extra_feats is None:
            return [torch.tensor([]) for _ in range(num_mols)]

        assert all([feat in batch.keys() for feat in self.extra_feats])

        feats = []
        for feat_name in self.extra_feats:
            feat_len = len(batch[feat_name]) // num_mols
            splits = [feat_len] * num_mols
            feat = list(torch.split(batch[feat_name] * FEAT_SCALING, splits))
            feats.append(feat)

        common_feats = []
        for j in range(len(feats[0])):
            common_feats.append([])
            for i in range(len(feats)):
                common_feats[-1].append(feats[i][j])
            common_feats[-1] = torch.cat(common_feats[-1])

        return common_feats

    def convolve(self, batch, xyz=None, xyz_grad=False, **kwargs):
        """

        Apply the convolutional layers to the batch.

        Args:
            batch (dict): dictionary of props

        Returns:
            r: new feature vector after the convolutions
            N: list of the number of atoms for each molecule in the batch
            xyz_grad (bool): whether we'll need the gradient wrt xyz
            xyz: xyz (with a "requires_grad") for the batch

        """

        # Note: we've given the option to input xyz from another source.
        # E.g. if you already created an xyz  and set requires_grad=True,
        # you don't want to make a whole new one.

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]
            xyz.requires_grad = xyz_grad

        r = batch["nxyz"][:, 0]
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

        return r, xyz

    def get_conf_fps(self, smiles_fp, mol_size):

        # total number of atoms
        num_atoms = smiles_fp.shape[0]
        # unmber of conformers
        num_confs = num_atoms // mol_size
        N = [mol_size] * num_confs
        conf_fps = []

        # split the atomic fingerprints up by conformer
        for atomic_fps in torch.split(smiles_fp, N):
            # sum them and then convert to molecular fp
            summed_atomic_fps = atomic_fps.sum(dim=0)
            mol_fp = self.mol_fp_nn(summed_atomic_fps)
            # add to the list of conformer fp's
            conf_fps.append(mol_fp)

        conf_fps = torch.stack(conf_fps)

        return conf_fps

    def post_process(self, batch, r, xyz, **kwargs):

        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        N = batch["num_atoms"].reshape(-1).tolist()
        num_confs = (torch.tensor(N) / torch.tensor(mol_sizes)).tolist()
        # split the fingerprints by species
        fps_by_smiles = torch.split(r, N)
        conf_fps_by_smiles = []

        for mol_size, smiles_fp in zip(mol_sizes, fps_by_smiles):
            conf_fps = self.get_conf_fps(smiles_fp, mol_size)
            conf_fps_by_smiles.append(conf_fps)

        # split the boltzmann weights by species
        boltzmann_weights = torch.split(batch["weights"], num_confs)

        # add extra features (e.g. from Morgan fingerprint or MPNN)
        extra_feats = self.add_features(batch=batch, **kwargs)

        # return everything in a dictionary
        outputs = dict(r=r,
                       N=N,
                       xyz=xyz,
                       conf_fps_by_smiles=conf_fps_by_smiles,
                       boltzmann_weights=boltzmann_weights,
                       mol_sizes=mol_sizes,
                       extra_feats=extra_feats)

        return outputs

    def make_embeddings(self, batch, xyz=None, **kwargs):

        r, xyz = self.convolve(batch=batch,
                               xyz=xyz,
                               **kwargs)
        outputs = self.post_process(batch=batch,
                                    r=r,
                                    xyz=xyz, **kwargs)

        return outputs, xyz

    def get_embeddings(self, batch, **kwargs):

        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        N = batch["num_atoms"].reshape(-1).tolist()
        num_confs = (torch.tensor(N) / torch.tensor(mol_sizes)).tolist()

        conf_fps_by_smiles = list(torch.split(batch["fingerprint"], num_confs))
        extra_feats = self.add_features(batch=batch, **kwargs)
        boltzmann_weights = torch.split(batch["weights"], num_confs)

        outputs = {"conf_fps_by_smiles": conf_fps_by_smiles,
                   "boltzmann_weights": boltzmann_weights,
                   "mol_sizes": mol_sizes,
                   "extra_feats": extra_feats}

        return outputs, None

    def pool(self, outputs):
        """

        Use the outputs of the convolutions to make a prediction.
        Here, the atomic fingerprints for each geometry get converted
        into a molecular fingerprint. Then, the molecular
        fingerprints for the different conformers of a given species
        get multiplied by the Boltzmann weights of those conformers and
        added together to make a final fingerprint for the species.
        Two fully-connected layers act on this final fingerprint to make
        a prediction.

        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        conf_fps_by_smiles = outputs["conf_fps_by_smiles"]
        batched_weights = outputs["boltzmann_weights"]
        mol_sizes = outputs["mol_sizes"]
        extra_feat_fps = outputs["extra_feats"]

        final_fps = []
        # go through each species
        for i in range(len(conf_fps_by_smiles)):
            boltzmann_weights = batched_weights[i]
            conf_fps = conf_fps_by_smiles[i]
            mol_size = mol_sizes[i]
            extra_feats = extra_feat_fps[i]

            # for backward compatibility
            if not hasattr(self, "boltz_nns"):
                self.boltz_nns = nn.ModuleList([self.boltz_nn])
            if not hasattr(self, "head_pool"):
                self.head_pool = "concatenate"

            # pool the atomic fingerprints
            final_fp = conf_pool(mol_size=mol_size,
                                 boltzmann_weights=boltzmann_weights,
                                 mol_fp_nn=self.mol_fp_nn,
                                 boltz_nns=self.boltz_nns,
                                 conf_fps=conf_fps,
                                 extra_feats=extra_feats,
                                 head_pool=self.head_pool)

            final_fps.append(final_fp)

        final_fps = torch.stack(final_fps)

        return final_fps

    def add_grad(self, batch, results, xyz):

        batch_keys = batch.keys()
        result_grad_keys = [key + "_grad" for key in results.keys()]
        for key in batch_keys:
            if key in result_grad_keys:
                base_result = results[key.replace("_grad", "")]
                results[key] = compute_grad(inputs=xyz,
                                            output=base_result)

        return results

    def forward(self, batch, xyz=None, **kwargs):

        if self.batch_embeddings:
            outputs, xyz = self.get_embeddings(batch, **kwargs)
        else:
            outputs, xyz = self.make_embeddings(batch, xyz, **kwargs)

        pooled_fp = self.pool(outputs)
        results = self.readout(pooled_fp)
        results = self.add_grad(batch=batch, results=results, xyz=xyz)

        return results
