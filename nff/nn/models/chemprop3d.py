
from munch import Munch

from chemprop.models import build_model as build_chemprop
from chemprop.data.data import MoleculeDataset


from nff.nn.models.conformers import WeightedConformers

import pdb

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

        WeightedConformers.__init__(self, modelparams)

        namespace = Munch(modelparams["chemprop"])
        self.cp_model = build_chemprop(namespace)

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

