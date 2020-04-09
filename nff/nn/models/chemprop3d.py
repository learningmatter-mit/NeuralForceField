from munch import Munch
import torch
import os

from chemprop.models import build_model as build_chemprop
from chemprop.data.data import MoleculeDataset


from nff.nn.models.conformers import WeightedConformers


class ChemProp3D(WeightedConformers):
    def __init__(self, modelparams):
        """
        Example:
            cp_params = {"num_tasks": 1, 
                         "dataset_type": "classification",
                         "atom_messages": False, 
                         "hidden_size": 300,
                         "bias": False, 
                         "depth": 3, 
                         "dropout": 0.2,
                         "undirected": False, 
                         "features_only": False,
                         "use_input_features": True,
                         "activation": "ReLU",
                         "features_dim": 1, # doesn't matter if loading
                         "ffn_num_layers": 2,
                         "ffn_hidden_size": 300,
                         "no_cache": False,
                         "cuda": True}

            modelparams.update({"chemprop": cp_params})
            model = ChemProp3D(modelparams)
        """

        cp_params = modelparams["chemprop"]
        cp_path = cp_params.get("transfer_cp")

        if cp_path is not None:
            cp_model, modelparams = self.init_with_cp(modelparams)
        else:
            namespace = Munch(cp_params)
            cp_model = build_chemprop(namespace)

        WeightedConformers.__init__(self, modelparams)
        self.cp_model = cp_model

    def load_cp_model(self, cp_params):

        cp_path = cp_params["transfer_cp"]
        gpu = cp_params["gpu"]

        model = torch.load(
            os.path.join(cp_path, 'best_model')
        )

        cp_model = model.cp_model
        cp_model.gpu = gpu
        cp_model = cp_model.to(gpu)

        cp_model.encoder.args.gpu = gpu
        cp_model.encoder.gpu = gpu
        cp_model.encoder = cp_model.encoder.to(gpu)

        cp_model.encoder.encoder.gpu = gpu

        for param in cp_model.parameters():
            param.requires_grad = False

        return cp_model

    def adjust_readout(self, modelparams, cp_params, cp_model):

        encoder = cp_model.encoder.encoder
        learned_cp_num = encoder.W_o.out_features
        extra_feats = cp_params["extra_features"]
        num_extra = sum([feat["length"]
                         for feat in extra_feats])
        cp_num = learned_cp_num + num_extra

        schnet_num = modelparams['mol_fp_layers'][-1]['param'
                                                      ]['out_features']

        for key, layers in modelparams['readoutdict'].items():
            layers[0]['param']['in_features'] = schnet_num + cp_num

    def init_with_cp(self, modelparams):

        cp_params = modelparams["chemprop"]
        cp_model = self.load_cp_model(cp_params=cp_params)
        self.adjust_readout(modelparams=modelparams,
                            cp_params=cp_params,
                            cp_model=cp_model)
        return cp_model, modelparams

    def get_chemprop_inp(self, batch, cp_data, smiles_dic):

        schnet_smiles = batch["smiles"]
        cp_idx = [smiles_dic[smiles] for smiles in schnet_smiles]
        cp_batch = MoleculeDataset([cp_data[idx]
                                    for idx in cp_idx])
        smiles_batch = cp_batch.smiles()
        features_batch = cp_batch.features()

        return (smiles_batch, features_batch)

    def add_features(self, batch, ex_data, smiles_dics, **kwargs):

        cp_data = ex_data[0]
        smiles_dic = smiles_dics[0]
        inputs = self.get_chemprop_inp(batch=batch,
                                       cp_data=cp_data,
                                       smiles_dic=smiles_dic)

        cp_feats = self.cp_model.encoder(*inputs)
        out_feats = [item for item in cp_feats]

        return out_feats
