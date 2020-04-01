from munch import Munch

from chemprop.data.utils import get_data
from chemprop.data.data import MoleculeDataset

from nff.train.trainer import Trainer


def hash_smiles(cp_data):
    cp_dataset = MoleculeDataset(cp_data)
    cp_smiles = cp_dataset.smiles()
    cp_smiles_dic = {smiles: i for i, smiles in enumerate(
        cp_smiles)}
    return cp_smiles_dic


def load_chemprop_data(features_path, smiles_path, args, **kwargs):
    data = get_data(path=smiles_path, features_path=features_path,
                    args=Munch(args))
    smiles_dic = hash_smiles(data)

    return data, smiles_dic


def load_external_data(dic):
    funcs = {"chemprop": load_chemprop_data}
    data_type = dic["data_type"]
    data, smiles_dic = funcs[data_type](**dic)
    return data, smiles_dic


class MixedDataTrainer(Trainer):
    def __init__(self, load_dics, *args, **kwargs):
        """
        Example:
                args =  Munch({"max_data_size": None, "features_dim": 2248,
                            "use_compound_names": False,
                            "features_generator": None})

            load_dics = [{"data_type": "chemprop",
                          "smiles_path": "/home/saxelrod/Repo/projects/coronavirus_data/data/mmff_train.csv",
                          "features_path": ["/home/saxelrod/Repo/projects/coronavirus_data/features/mmff_train.npz"],
                          "args": args,
                          "name": "chemprop_data"}]
        """

        Trainer.__init__(self, *args, **kwargs)
        self.external_data = []
        self.smiles_dics = []

        for load_dic in load_dics:
            data, smiles_dic = load_external_data(load_dic)
            self.external_data.append(data)
            self.smiles_dics.append(smiles_dic)

    def call_model(self, batch):
        output = self._model(batch=batch,
                             ex_data=self.external_data,
                             smiles_dics=self.smiles_dics)
        return output
