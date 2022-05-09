import torch
import torch.nn as nn

from nff.nn.layers import DEFAULT_DROPOUT_RATE
from nff.nn.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
    ConfAttention,
    LinearConfAttention
)
from nff.nn.graphop import conf_pool
from nff.nn.utils import construct_sequential
from nff.utils.scatter import compute_grad
from nff.utils.confs import split_batch


class WeightedConformers(nn.Module):
    """
    Model that uses a representation of a molecule in terms of different 3D
    conformers to predict properties. The fingerprints of each conformer are 
    generated using the SchNet model.
    """

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
        self.extra_feats = modelparams.get("extra_features")
        self.ext_feat_types = modelparams.get("ext_feat_types")

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

        # whether this is a classifier
        self.classifier = modelparams["classifier"]

        # whether to embed fingerprints or just use external features
        self.use_mpnn = modelparams.get("use_mpnn", True)

        # How to aggregate the node features when making the molecular
        # fingerprint

        self.pool_type = modelparams.get("pool_type", "sum")

    def make_boltz_nn(self, boltzmann_dict):
        """
        Make the section of the network that creates weights for each
        conformer, which may or may not be equal to the statistical
        boltzmann weights.
        Args:
            boltzmann_dict (dict): dictionary with information about
                this section of the network.
        Returns:
            networks (nn.ModuleList): list of networks that get applied
                to the conformer fingerprints to aggregate them. If
                it contains more than one network, the different fingerprints
                produced will either be averaged or concatenated at the end.
        """

        networks = nn.ModuleList([])

        # if you just want to multiply the boltmzann weight by each conformer
        # fingerprint, return nothing

        if boltzmann_dict["type"] == "multiply":
            return [None]

        # if you supply a dictionary of type `layers`, then the dictionary
        # under the key `layers` will be used to create the corresponding
        # network

        elif boltzmann_dict["type"] == "layers":
            layers = boltzmann_dict["layers"]
            networks.append(construct_sequential(layers))

        # if you ask for some sort of attention network, then make one such
        # network for each of the number of heads

        elif "attention" in boltzmann_dict["type"]:

            if boltzmann_dict["type"] == "attention":
                module = ConfAttention
            elif boltzmann_dict["type"] == "linear_attention":
                module = LinearConfAttention
            else:
                raise NotImplementedError

            # how many attention heads
            num_heads = boltzmann_dict.get("num_heads", 1)
            # whether to just use equal weights and not learnable weights
            # (useful for ablation studies)
            equal_weights = boltzmann_dict.get("equal_weights", False)
            # what function to use to convert the alpha_ij to probabilities
            prob_func = boltzmann_dict.get("prob_func", 'softmax')

            # add a network for each head
            for _ in range(num_heads):

                mol_basis = boltzmann_dict["mol_basis"]
                boltz_basis = boltzmann_dict["boltz_basis"]
                final_act = boltzmann_dict["final_act"]

                networks.append(module(mol_basis=mol_basis,
                                       boltz_basis=boltz_basis,
                                       final_act=final_act,
                                       equal_weights=equal_weights,
                                       prob_func=prob_func))

        return networks

    def add_features(self, batch, **kwargs):
        """
        Get any extra per-species features that were requested for 
        the dataset. 
        Args:
            batch (dict): batched sample of species
        Returns:
            feats (list): list of feature tensors for each species.
        """

        N = batch["num_atoms"].reshape(-1).tolist()
        num_mols = len(N)

        # if you didn't ask for any extra features, or none of the requested
        # features are per-species features, return empty tensors

        if self.extra_feats is None or "species" not in self.ext_feat_types:
            return [torch.tensor([]) for _ in range(num_mols)]

        assert all([feat in batch.keys() for feat in self.extra_feats])
        feats = []

        # go through each extra per-species feature

        for feat_name, feat_type in zip(self.extra_feats, self.ext_feat_types):

            if feat_type == "conformer":
                continue

            # how long each feature is
            feat_len = len(batch[feat_name]) // num_mols
            # split the batched features up by species and add them
            # to the list
            splits = [feat_len] * num_mols
            feat = torch.stack(list(
                torch.split(batch[feat_name], splits)))
            feats.append(feat)

        # concatenate the features
        feats = torch.cat(feats, dim=-1)

        return feats

    def convolve_sub_batch(self,
                           batch,
                           xyz=None,
                           xyz_grad=False,
                           **kwargs):
        """

        Apply the convolutional layers to a sub-batch.

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
        # to deal with any shape mismatches
        if hasattr(offsets, 'max') and offsets.max() == 0:
            offsets = 0

        if "distances" in batch:
            e = batch["distances"][:, None]
        else:
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

        # for backwards compatability
        if not hasattr(self, "classifier"):
            self.classifier = True

        # split batches as necessary
        if sub_batch_size is None:
            sub_batches = [batch]
        else:
            sub_batches = split_batch(batch, sub_batch_size)

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

    def get_external_3d(self,
                        batch,
                        n_conf_list):
        """
        Get any extra 3D per-conformer features that were requested for 
        the dataset. 
        Args:
            batch (dict): batched sample of species
            n_conf_list (list[int]): list of number of conformers in each 
                species.
        Returns:
            split_extra (list): list of stacked per-cofnormer feature tensors 
                for each species.
        """

        # if you didn't ask for any extra features, or none of the requested
        # features are per-conformer features, return empty tensors

        if (self.extra_feats is None or
                "conformer" not in self.ext_feat_types):
            return

        # get all the features and split them up by species

        extra_conf_fps = []
        for feat_name, feat_type in zip(self.extra_feats,
                                        self.ext_feat_types):
            if feat_type == "conformer":
                extra_conf_fps.append(batch[feat_name])

        extra_conf_fps = torch.cat(extra_conf_fps, dim=-1)
        split_extra = torch.split(extra_conf_fps, n_conf_list)

        return split_extra

    def get_conf_fps(self,
                     smiles_fp,
                     mol_size,
                     batch,
                     split_extra,
                     idx):
        """
        Get per-conformer fingerprints.
        Args:
            smiles_fp (torch.Tensor): per-atom fingerprints
                for every atom in the species. Note that this
                has length mol_size x n_confs, where `mol_size`
                is the number of atoms in the molecule, and
                `n_confs` is the number of conformers.
            mol_size (int): Number of atoms in the molecule
            batch (dict): batched sample of species
            split_extra (list): extra 3D fingerprints split by
                species 
            idx (int): index of the current species in the batch.
        """

        # total number of atoms
        num_atoms = smiles_fp.shape[0]
        # unmber of conformers
        num_confs = num_atoms // mol_size
        N = [mol_size] * num_confs
        conf_fps = []

        if getattr(self, "pool_type", None) is None:
            self.pool_type = "sum"

        # split the atomic fingerprints up by conformer
        for atomic_fps in torch.split(smiles_fp, N):
            # sum them and then convert to molecular fp
            if self.pool_type == 'sum':
                summed_atomic_fps = atomic_fps.sum(dim=0)
            elif self.pool_type == 'mean':
                summed_atomic_fps = atomic_fps.mean(dim=0)
            else:
                raise NotImplementedError

            # put them through the network to convert summed
            # atomic fps to a molecular fp
            mol_fp = self.mol_fp_nn(summed_atomic_fps)
            # add to the list of conformer fps
            conf_fps.append(mol_fp)

        # stack the conformer fps
        conf_fps = torch.stack(conf_fps)

        # if there are any extra 3D fingerprints, add them here
        if split_extra is not None:
            this_extra = split_extra[idx]
            conf_fps = torch.cat([conf_fps, this_extra], dim=-1)

        return conf_fps

    def post_process(self,
                     batch,
                     r,
                     xyz,
                     **kwargs):
        """
        Split various items up by species, convert atomic fingerprints
        to molecular fingerprints, and incorporate non-learnable features.
        Args:
            batch (dict): batched sample of species
            r (torch.Tensor): atomwise learned features from the convolutions
            xyz (torch.Tensor): xyz of the batch
        Returns:
            output (dict): various new items
        """

        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        N = batch["num_atoms"].reshape(-1).tolist()
        num_confs = (torch.tensor(N) / torch.tensor(mol_sizes)).long().tolist()
        # split the fingerprints by species
        fps_by_smiles = torch.split(r, N)
        # get extra 3D fingerprints
        split_extra = self.get_external_3d(batch,
                                           num_confs)

        # get all the conformer fingerprints for each species
        conf_fps_by_smiles = []
        for i, smiles_fp in enumerate(fps_by_smiles):
            conf_fps = self.get_conf_fps(smiles_fp=smiles_fp,
                                         mol_size=mol_sizes[i],
                                         batch=batch,
                                         split_extra=split_extra,
                                         idx=i)

            conf_fps_by_smiles.append(conf_fps)

        # split the boltzmann weights by species
        boltzmann_weights = torch.split(batch["weights"], num_confs)

        # add any extra per-species features
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

    def fps_no_mpnn(self, batch, **kwargs):
        """
        Get fingerprints without using an MPNN to get any learned fingerprints.
        Args:
            batch (dict): batched sample of species
        Returns:
            output (dict): various new items
        """

        # number of atoms in each species, which is greater than `mol_size`
        # if the number of conformers exceeds 1
        N = batch["num_atoms"].reshape(-1).tolist()
        # number of atoms in each molecule
        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        # number of conformers per species
        n_conf_list = (torch.tensor(N) / torch.tensor(mol_sizes)).tolist()

        # get the conformer fps for each smiles
        conf_fps_by_smiles = self.get_external_3d(batch,
                                                  n_conf_list)

        # add any per-species fingerprints
        boltzmann_weights = torch.split(batch["weights"], n_conf_list)
        extra_feats = self.add_features(batch=batch, **kwargs)

        outputs = {"conf_fps_by_smiles": conf_fps_by_smiles,
                   "boltzmann_weights": boltzmann_weights,
                   "mol_sizes": mol_sizes,
                   "extra_feats": extra_feats}

        return outputs

    def make_embeddings(self,
                        batch,
                        xyz=None,
                        **kwargs):
        """
        Make all conformer fingerprints.
        Args:
            batch (dict): batched sample of species
            xyz (torch.Tensor): xyz of the batch
        Returns:
            output (dict): various new items
            xyz (torch.Tensor): xyz of the batch
        """

        # for backward compatability
        if not hasattr(self, "use_mpnn"):
            self.use_mpnn = True

        # if using an MPNN, apply the convolution layers
        # and then post-process
        if self.use_mpnn:
            r, xyz = self.convolve(batch=batch,
                                   xyz=xyz,
                                   **kwargs)
            outputs = self.post_process(batch=batch,
                                        r=r,
                                        xyz=xyz,
                                        **kwargs)

        # otherwise just use the non-learnable features
        else:
            outputs = self.fps_no_mpnn(batch, **kwargs)
            xyz = None

        return outputs, xyz

    def pool(self, outputs):
        """

        Pool the per-conformer outputs of the convolutions.
        Here, the atomic fingerprints for each geometry get converted
        into a molecular fingerprint. Then, the molecular
        fingerprints for the different conformers of a given species
        get multiplied by the Boltzmann weights or learned weights of 
        those conformers and added together to make a final fingerprint 
        for the species.

        Args:
            batch (dict): dictionary of props

        Returns:
            final_fps (torch.Tensor): final per-species fingerprints
            final_weights (list): weights assigned to each conformer
                in the ensemble.

        """

        conf_fps_by_smiles = outputs["conf_fps_by_smiles"]
        batched_weights = outputs["boltzmann_weights"]
        mol_sizes = outputs["mol_sizes"]
        extra_feat_fps = outputs["extra_feats"]

        final_fps = []
        final_weights = []

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
            final_fp, learned_weights = conf_pool(
                mol_size=mol_size,
                boltzmann_weights=boltzmann_weights,
                mol_fp_nn=self.mol_fp_nn,
                boltz_nns=self.boltz_nns,
                conf_fps=conf_fps,
                head_pool=self.head_pool)

            # add extra features if there are any
            if extra_feats is not None:
                extra_feats = extra_feats.to(final_fp.device)
                final_fp = torch.cat((final_fp, extra_feats))

            final_fps.append(final_fp)
            final_weights.append(learned_weights)

        final_fps = torch.stack(final_fps)

        return final_fps, final_weights

    def add_grad(self, batch, results, xyz):
        """
        Add any required gradients of the predictions.
        Args:
            batch (dict): dictionary of props
            results (dict): dictionary of predicted values
            xyz (torch.tensor): (optional) coordinates
        Returns:
            results (dict): results updated with any gradients
                requested.
        """

        batch_keys = batch.keys()
        # names of the gradients of each property
        result_grad_keys = [key + "_grad" for key in results.keys()]
        for key in batch_keys:
            # if the batch with the ground truth contains one of
            # these keys, then compute its predicted value
            if key in result_grad_keys:
                base_result = results[key.replace("_grad", "")]
                results[key] = compute_grad(inputs=xyz,
                                            output=base_result)

        return results

    def forward(self,
                batch,
                xyz=None,
                **kwargs):
        """
        Call the model.
        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates
        Returns:
            results (dict): dictionary of predicted values
        """

        # for backwards compatibility
        if not hasattr(self, "classifier"):
            self.classifier = True

        # make conformer fingerprints
        outputs, xyz = self.make_embeddings(batch, xyz, **kwargs)
        # pool the fingerprints
        pooled_fp, final_weights = self.pool(outputs)
        # apply network to fingerprints get predicted value
        results = self.readout(pooled_fp)

        # add sigmoid if it's a classifier and not in training mode
        if self.classifier and not self.training:
            keys = list(self.readout.readout.keys())
            for key in keys:
                results[key] = torch.sigmoid(results[key])

        # add any required gradients
        results = self.add_grad(batch=batch, results=results, xyz=xyz)
        # add in the weights of each conformer for later analysis
        results.update({"learned_weights": final_weights})

        return results
