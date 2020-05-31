import torch.nn as nn

from nff.nn.modules import NodeMultiTaskReadOut

from munch import Munch

# from chemprop.models import build_model as build_chemprop
from chemprop.data.data import MoleculeDataset


class ChemProp2D(nn.Module):

    """
    A wrapper around regular ChemProp, which acts only on graphs and not 3D strucutres.
    This wrapper exists so that ChemProp can be trained in the same way as any other NFF model, 
    like SchNet or ChemProp3D. 
    """

    def __init__(self, modelparams):

        nn.Module.__init__(self)
        readoutdict = modelparams["readoutdict"]
        # the readout acts on this final molceular fp
        self.readout = NodeMultiTaskReadOut(multitaskdict=readoutdict)

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

        return cp_feats

    def forward(self, batch, xyz=None, **kwargs):

        N = batch["num_atoms"].reshape(-1).tolist()
        num_mols = len(N)
        cp_feats = self.add_features(batch=batch, num_mols=num_mols,
                                     **kwargs)
        results = self.readout(cp_feats)

        return results
