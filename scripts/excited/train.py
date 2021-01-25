import torch
import copy
import numpy as np
import random
import os
import json

from rdkit import Chem
from sklearn.model_selection import train_test_split


from nff.data.loader import BalancedFFSampler
from nff.data import Dataset, split_train_validation_test


def sampler_and_rmsds(sampler_kwargs):

    sampler = BalancedFFSampler(**sampler_kwargs)

    balance_dict = sampler.balance_dict
    cluster_rmsds = balance_dict["cluster_rmsds"]

    return sampler, cluster_rmsds


def add_diabat(cluster_rmsds,
               max_rmsd,
               num_states,
               assignments,
               dset,
               diabatic_keys):
    """
    Example:
        Here 0 is cis and 1 is trans:
            assignments = {0: ["energy_0", "energy_1"],
                           1: ["energy_1", "energy_0"]}

    """

    # props =

    # is this the right order of the indexing?
    ref_rmsds = cluster_rmsds[:, :num_states]

    # the diabatic states that each geom is closest to
    closest_refs = ref_rmsds.argmin(-1)
    # whether it's really close enough for an adiabatic
    # state to be considered equal to the diabatic state
    are_diabats = ref_rmsds.min(-1) <= max_rmsd

    # assignment of diabatic energies
    diabat_props = {key: [] for key in diabatic_keys}

    for i, batch in enumerate(dset):

        closest_ref = closest_refs[i]
        is_diabat = are_diabats[i]
        adiabats = assignments[int(closest_ref)]

        if is_diabat:
            for diabat_key, adiabat in zip(diabatic_keys, adiabats):
                diabat_props[diabat_key].append(batch[adiabat])
        else:
            nan = batch[adiabats[0]] * float('nan')
            for diabat_key in diabatic_keys:
                diabat_props[diabat_key].append(nan)

    for key, val in diabat_props.items():
        diabat_props[key] = torch.stack(val).reshape(-1)

    dset.props.update(diabat_props)

    return dset


def rm_stereo(smiles):
    new_smiles = smiles.replace("\\", "").replace("/", "")
    return new_smiles


def to_rdmol(smiles,
             add_hs=False,
             keep_stereo=False):

    this_smiles = copy.deepcopy(smiles)
    if not keep_stereo:
        this_smiles = rm_stereo(this_smiles)

    mol = Chem.MolFromSmiles(this_smiles)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if add_hs:
        mol = Chem.AddHs(mol)

    return mol


def substrucs_to_mols(smiles_list,
                      keep_stereo=False):
    mols = []
    for sub_list in smiles_list:
        these_mols = []
        for smiles in sub_list:
            mol = to_rdmol(smiles=smiles,
                           add_hs=False,
                           keep_stereo=keep_stereo)
            these_mols.append(mol)
        mols.append(these_mols)
    return mols


def sub_mol_present(batch_mol,
                    substruc_mols):

    present = False
    for mol_list in substruc_mols:
        are_substrucs = []
        for ref_mol in mol_list:
            is_substruc = batch_mol.HasSubstructMatch(ref_mol)
            are_substrucs.append(is_substruc)

        # "and" comparison
        is_match = all(are_substrucs)

        # "or" comparison
        if is_match:
            present = True
            break

    return present


def prune_by_substruc(dset,
                      substruc_smiles,
                      keep_stereo=False):
    """
    Given a substructure that must be present, prune the dataset to remove geoms
    that don't have it.

    Args:
        substruc_smiles (list[list]): the SMILES of the substructures that must
            be present. Organized in the form [[sub_0, sub_1],
            [sub_2, sub_3, sub_4], [sub_1, sub_2]], which is parsed as
            ((sub_0 and sub_1) or (sub_2 and sub_3 and sub_4) or (sub_1 and sub_3))
        keep_stereo (bool): if True, the substructure in its exact stereochemical
            form must be present in a species. If False, only the substructure
            without stereochemical information must be present.

    Example:

    """

    keep_idx = []
    substruc_mols = substrucs_to_mols(smiles_list=substruc_smiles,
                                      keep_stereo=keep_stereo)

    for i, batch in enumerate(dset):
        batch_mol = to_rdmol(batch['smiles'],
                             add_hs=False,
                             keep_stereo=keep_stereo)
        keep = sub_mol_present(batch_mol=batch_mol,
                               substruc_mols=substruc_mols)
        if keep:
            keep_idx.append(i)

    dset.change_idx(keep_idx)

    return dset


def analyze_split(these_weights, num_chunks):

    total_weight = these_weights.sum()
    expec_weight = 1 / num_chunks

    print("Chunk weight: %.3f" % total_weight)
    print("Expected weight: %.3f" % expec_weight)


def make_parallel_chunks(dset, num_chunks, sampler):

    sample_split = np.array_split(list(range(len(dset))), num_chunks)
    geoms_in_splits = [len(i) for i in sample_split]
    weights = copy.deepcopy(sampler.balance_dict["weights"]).reshape(-1)

    dset_chunks = []

    for geoms_per_split in geoms_in_splits:

        new_props = {}
        for key, val in dset.props.items():
            new_props[key] = copy.deepcopy(val[:geoms_per_split])
            dset.props[key] = val[geoms_per_split:]

        these_weights = weights[:geoms_per_split]
        weights = weights[geoms_per_split:]
        analyze_split(these_weights=these_weights,
                      num_chunks=num_chunks)

        dset_chunk = Dataset(props=new_props, check_props=False)
        chunk_dic = {"dset": dset_chunk, "weights": these_weights}
        dset_chunks.append(chunk_dic)

    return dset_chunks


def load_dset(job_info):
    base_dset_path = job_info["base_dset_path"]
    dset = Dataset.from_file(base_dset_path)

    return dset


def save_sample_weights(sampler, job_info):

    weights = sampler.balance_dict["weights"].reshape(-1).tolist()
    model_path = job_info["model_path"]
    save_path = os.path.join(model_path, "sample_weights.json")

    with open(save_path, "w") as f_open:
        json.dump(weights, f_open)

    return save_path


def make_split(dset, job_info):
    """
    Need to make sure the holdouts go in a different split than the
    non-holdouts, if we do the holdout thing.
    """

    splits = split_train_validation_test(dset,
                                         val_size=job_info["val_size"],
                                         test_size=job_info["test_size"],
                                         seed=job_info["seed"])

    split_keys = ["train", "validation", "test"]
    split_dic = {key: {"dset": split}
                 for key, split in zip(split_keys, splits)}

    return split_dic


def split_and_sample(dset, job_info):

    split_dic = make_split(dset=dset, job_info=job_info)

    for key, sub_dic in split_dic.items():

        this_dset = sub_dic["dset"]
        sampler_kwargs = {"props": this_dset.props,
                          **job_info["sampler_kwargs"]}
        sampler, cluster_rmsds = sampler_and_rmsds(
            sampler_kwargs=sampler_kwargs)
        this_dset = add_diabat(cluster_rmsds=cluster_rmsds,
                               max_rmsd=job_info["max_rmsd"],
                               num_states=job_info["num_states"],
                               assignments=job_info["assignments"],
                               dset=this_dset,
                               diabatic_keys=job_info["diabatic_keys"])

        split_dic[key] = {"dset": this_dset, "sampler": sampler}
    return split_dic


def save_as_chunks(split_dic, job_info):

    model_path = job_info["model_path"]
    dset_chunks = {split: make_parallel_chunks(
        dset=sub_dic["dset"],
        num_chunks=job_info["num_parallel"],
        sampler=sub_dic["sampler"]) for split, sub_dic in split_dic.items()}

    for split, chunk_dics in dset_chunks.items():
        for i, chunk_dic in enumerate(chunk_dics):

            direc = os.path.join(model_path, str(i))
            if not os.path.isdir(direc):
                os.makedirs(direc)

            dset_chunk = chunk_dic["dset"]
            dset_path = os.path.join(direc, f"{split}.pth.tar")
            dset_chunk.save(dset_path)

            weights = chunk_dic["weights"].reshape(-1).tolist()
            weight_path = os.path.join(direc, f"{split}_weights.json")
            with open(weight_path, "w") as f_open:
                json.dump(weights, f_open)


def idx_in_smiles_list(dset, specs, present):
    if present:
        idx = [i for i, batch in enumerate(dset)
               if batch['smiles'] in specs]
    else:
        idx = [i for i, batch in enumerate(dset)
               if batch['smiles'] not in specs]
    return idx


def norm_test_size(train_name, test_name, split_sizes):
    norm_test = split_sizes[test_name] / (split_sizes[test_name]
                                          + split_sizes[train_name])
    return norm_test


def one_preset_split(dset,
                     present_dic,
                     spec_dic,
                     split_sizes,
                     seed):

    # the one split that's present
    present_split = [key for key, val in present_dic.items()
                     if val][0]
    # the two that are missing
    missing_splits = [key for key, val in present_dic.items()
                      if not val]

    # for one specified we get the indices by matching the smiles
    # to the requested smiles in `spec_dic`
    present_idx = idx_in_smiles_list(dset,
                                     specs=spec_dic[present_split],
                                     present=True)

    # for the two missing we get every geom whose smiles isn't in
    #  `spec_dic[present_split]` (specified by present=False)
    missing_idx = idx_in_smiles_list(dset,
                                     specs=spec_dic[present_split],
                                     present=False)

    # the normalized size of the second split without species

    missing_size_1 = norm_test_size(train_name=missing_splits[0],
                                    test_name=missing_splits[1],
                                    split_sizes=split_sizes)

    # the indices of the missing splits
    missing_idx_0, missing_idx_1 = train_test_split(
        missing_idx,
        test_size=missing_size_1,
        random_state=seed)

    # assign to a dictionary
    split_idx = {present_split: present_idx,
                 missing_splits[0]: missing_idx_0,
                 missing_splits[1]: missing_idx_1}

    return split_idx


def two_preset_split(dset,
                     present_dic,
                     spec_dic,
                     split_sizes,
                     seed):

    # the two splits that are present
    present_splits = [key for key, val in present_dic.items()
                      if val]
    # the one that's missing
    missing_split = [key for key, val in present_dic.items()
                     if not val][0]

    # the two that are present just get matched by species
    split_idx = {split: idx_in_smiles_list(dset,
                                           specs=spec_dic[split],
                                           present=True)
                 for split in present_splits}

    # the one that's not present gets anti-matched by the union
    # of the two present sets of species

    exclude_specs = (spec_dic[present_splits[0]]
                     + spec_dic[present_splits[1]])
    split_idx.update({missing_split:
                      idx_in_smiles_list(dset,
                                         specs=exclude_specs,
                                         present=False)})

    return split_idx


def three_preset_split(dset, spec_dic):
    # if all three are present then they all just get matched
    split_idx = {split: idx_in_smiles_list(dset,
                                           specs=specs,
                                           present=True)
                 for split, specs in spec_dic.items()}
    return split_idx


def split_by_species(dset,
                     species_splits,
                     split_sizes,
                     seed):
    """
    Split the dataset given a requested assignment
    of each species to a particular split. Not all splits
    have to be specified; for example, you can specify
    that `test` must consist of a set of specific species,
    and then `train` and `val` will be made by splitting
    all the remaining species accoridng to `split_sizes`.

    Args:
        dset (nff.data.Dataset): NFF dataset
        species_splits (dict): dictionary that specifies which
            splits have which species. Has, for example, the form
            {"test": ["C=N"], "val": ["C#N"], "train": ["CC", "CCC"]}.
        split_sizes (dict): dictionary that, for example, has
             the form {"test": 0.2, "train": 0.5, "val": 0.3}.
             You only have to specify the sizes of the splits
             whose species you don't specify. For example,
             if you specify the species for `test` and not
             for the other two, you only need to have
             `split_sizes = {"train": 0.5, "val": 0.3}`,
             or `split_sizes = {"train": 0.625, "val": 0.375}`,
             which are equivalent since the split sizes get
             normalized.
        seed (int): seed for the random splitting.
    Returns:
        train_idx (list): indices of geoms in the training set
        val_idx (list): indices of geoms in the validation set
        test_idx (list): indices of geoms in the test set
    """

    names = ["train", "val", "test"]
    # dictionary where each key is a split, and each value is
    # whether or not the species were specified for that split
    present_dic = {name: name in species_splits for name in names}

    # dictionary where each key is a split, and each value is
    # the species associated with that split (empty list if the species
    # weren't specified)
    spec_dic = {name: species_splits.get(name, []) for name in names}

    if sum(present_dic.values()) == 1:
        split_idx = one_preset_split(dset=dset,
                                     present_dic=present_dic,
                                     spec_dic=spec_dic,
                                     split_sizes=split_sizes,
                                     seed=seed)

    elif sum(present_dic.values()) == 2:
        split_idx = two_preset_split(dset=dset,
                                     present_dic=present_dic,
                                     spec_dic=spec_dic,
                                     split_sizes=split_sizes,
                                     seed=seed)

    elif sum(present_dic.values()) == 3:
        split_idx = three_preset_split(dset=dset, spec_dic=spec_dic)

    # extract split indices from the dictionary
    train_idx = split_idx["train"]
    val_idx = split_idx["val"]
    test_idx = split_idx["test"]

    return (train_idx, val_idx, test_idx)


def diabat_and_weights(dset, job_info):

    dset = prune_by_substruc(dset=dset,
                             substruc_smiles=job_info["substruc_smiles"],
                             keep_stereo=job_info["keep_stereo"])
    dset.shuffle()

    split_dic = split_and_sample(dset=dset, job_info=job_info)
    save_as_chunks(split_dic=split_dic,
                   job_info=job_info)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main(seed):
    set_seed(seed)
