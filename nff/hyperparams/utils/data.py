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


def trim_confs(dataset, num_confs):

    for i, nxyz in enumerate(dataset.props["nxyz"]):

        mol_size = dataset.props["mol_size"][i]
        new_num_atoms = int(mol_size * num_confs)

        if new_num_atoms == dataset.props["num_atoms"][i]:
        	continue
        
        # update the number of atoms and the nxyz
        dataset.props["num_atoms"][i] = new_num_atoms
        dataset.props["nxyz"][i] = dataset.props["nxyz"][i][
            :new_num_atoms]

        # trim the weights
        new_weights = dataset.props["weights"][i][:num_confs]
        new_weights /= new_weights.sum()
        dataset.props["weights"][i] = new_weights

        # trim the neighbour list
        new_nbr_list = dataset.props["nbr_list"][i]
        good_idx = []
        for j, pair in enumerate(new_nbr_list):
        	max_idx = np.max(pair.numpy())
        	if max_idx <= new_num_atoms - 1:
        		good_idx.append(j)

        new_nbr_list = new_nbr_list[good_idx]
        dataset.props["nbr_list"][i] = new_nbr_list



def get_data_dic(base_train, base_val, base_test, params):

    data_dic = {"train": {"dataset": base_train.copy()},
                "val": {"dataset": base_val.copy()},
                "test": {"dataset": base_test.copy()}}

    for key, split in data_dic.items():
        this_set = split["dataset"]
        if params.get("morgan_length") is not None:
            add_morgan(dataset=this_set,
                       vec_length=params["morgan_length"])
        if params.get("num_confs") is not None:
            num_confs = params["num_confs"]
            trim_confs(dataset=this_set,
                       num_confs=num_confs)
        loader = DataLoader(this_set,
                            batch_size=params["batch_size"],
                            collate_fn=collate_dicts)
        data_dic[key]["loader"] = loader
    return data_dic
