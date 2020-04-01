from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np

from torch.utils.data import DataLoader
from nff.data import collate_dicts


def add_morgan(dataset, vec_length):
    dataset.props["morgan"] = []
    for smiles in dataset.props['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if vec_length != 0:
            morgan = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=vec_length)
        else:
            morgan = []

        # shouldn't be a long tensor if we're going
        # to apply an NN to it

        arr_morgan = np.array(list(morgan)).astype('float32')
        morgan_tens = torch.tensor(arr_morgan)
        dataset.props["morgan"].append(morgan_tens)

def add_features(params, dataset):
    func_dic = {"morgan": add_morgan}
    extra_features = params["extra_features"]

    for dic in extra_features:
        name = dic["name"]
        vec_length = dic["length"]
        func_dic[name](dataset, vec_length)
        
def trim_confs(dataset, num_confs):

    for i in range(len(dataset)):

        mol_size = dataset.props["mol_size"][i]
        old_num_atoms = dataset.props["num_atoms"][i]

        # if conformers in batch beforehand
        # are less than num_confs, use
        # the old number of conformers

        confs_in_batch = old_num_atoms // mol_size
        new_num_atoms = int(mol_size * min(
            confs_in_batch, num_confs))
        dataset.props["num_atoms"][i] = torch.tensor(
            new_num_atoms)

        # update nxyz
        dataset.props["nxyz"][i] = dataset.props["nxyz"][i][
            :new_num_atoms]

        # trim the weights
        new_weights = dataset.props["weights"][i][:num_confs]
        new_weights /= new_weights.sum()
        dataset.props["weights"][i] = new_weights

        # trim the neighbour list

        nbr_list = dataset.props["nbr_list"][i]
        max_neighbor, _ = torch.max(nbr_list, dim=1)
        mask = (max_neighbor <= new_num_atoms - 1)
        good_idx = mask.nonzero().reshape(-1)

        dataset.props["nbr_list"][i] = nbr_list[good_idx]

def get_data_dic(base_train, base_val, base_test, params):

    data_dic = {"train": {"dataset": base_train.copy()},
                "val": {"dataset": base_val.copy()},
                "test": {"dataset": base_test.copy()}}

    for key, split in data_dic.items():
        this_set = split["dataset"]
        if params.get("extra_features") is not None:
            add_features(params=params,
                         dataset=this_set)
        if params.get("num_confs") is not None:
            num_confs = params["num_confs"]
            trim_confs(dataset=this_set,
                       num_confs=num_confs)
        loader = DataLoader(this_set,
                            batch_size=params["batch_size"],
                            collate_fn=collate_dicts)
        data_dic[key]["loader"] = loader
    return data_dic
