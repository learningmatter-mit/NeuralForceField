import django
django.setup()

from nff.utils.misc import tqdm_enum
from nff.data import Dataset, split_train_validation_test
from nff.data.loader import BalancedFFSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit import Chem
import numpy as np
import torch
import argparse
import json
import os
import random
import copy
from neuralnet.utils.data import convg_and_ci_geoms, equil_geoms


# if you sub-divide your job_info with these names, then their contents
# will be read in. Any other division will be left as is

ALLOWED_DIVIDER_NAMES = ["pruning", "balanced_sampling", "diabatization",
                         "splitting"]


def get_ref_dic(ref_config_type, **kwargs):

    if ref_config_type == "convg_and_ci":
        ref_dic = convg_and_ci_geoms(**kwargs)
    elif ref_config_type == "equil":
        ref_dic = equil_geoms(**kwargs)
    else:
        raise NotImplementedError

    return ref_dic


def sampler_and_rmsds(balance_type, sampler_kwargs):

    sampler = BalancedFFSampler(balance_type=balance_type,
                                **sampler_kwargs)

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
            assignments = {"0": ["energy_0", "energy_1"],
                           "1": ["energy_1", "energy_0"]}

    """

    # props =

    # is this the right order of the indexing?
    ref_rmsds = cluster_rmsds[:, :num_states]

    # the diabatic states that each geom is closest to
    closest_refs = ref_rmsds.argmin(-1)
    # whether it's really close enough for an adiabatic
    # state to be considered equal to the diabatic state

    are_diabats = ref_rmsds.min(-1)[0] <= max_rmsd

    # assignment of diabatic energies
    diag_diabat = (np.array(diabatic_keys)
                   .diagonal().reshape(-1)
                   .tolist())
    diabat_props = {key: [] for key in diag_diabat}

    for i, batch in enumerate(dset):

        closest_ref = closest_refs[i]
        is_diabat = are_diabats[i]
        adiabats = assignments[str(closest_ref.item())]

        if is_diabat:
            for diabat_key, adiabat in zip(diag_diabat, adiabats):
                diabat_props[diabat_key].append(batch[adiabat])
        else:
            nan = batch[adiabats[0]] * float('nan')
            for diabat_key in diag_diabat:
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
                    substruc_mols,
                    use_chirality):

    present = False
    for mol_list in substruc_mols:
        are_substrucs = []
        for ref_mol in mol_list:
            is_substruc = batch_mol.HasSubstructMatch(
                ref_mol,
                useChirality=use_chirality)
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

    if not substruc_smiles:
        return dset

    keep_idx = []
    substruc_mols = substrucs_to_mols(smiles_list=substruc_smiles,
                                      keep_stereo=keep_stereo)

    for i, batch in enumerate(dset):
        batch_mol = to_rdmol(batch['smiles'],
                             add_hs=False,
                             keep_stereo=keep_stereo)
        keep = sub_mol_present(batch_mol=batch_mol,
                               substruc_mols=substruc_mols,
                               use_chirality=keep_stereo)
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


def get_model_path(job_info):
    if "model_path" in job_info:
        return job_info["model_path"]

    weightpath = job_info["weightpath"]
    if not os.path.isdir(weightpath):
        weightpath = job_info["mounted_weightpath"]
    model_path = os.path.join(weightpath, job_info["nnid"])
    return model_path


def to_json(dic):
    for key, val in dic.items():
        if isinstance(val, dict):
            dic[key] = to_json(val)
        else:
            if hasattr(val, "tolist"):
                dic[key] = val.tolist()
            else:
                dic[key] = val
    return dic




def make_random_split(dset, split_sizes, seed):

    frac_split_sizes = {}
    for name in ["val", "test"]:
        size = copy.deepcopy(split_sizes[name])

        # If it's greater than one then that means
        # it's telling you the total number to use,
        # not the percentage. In this case divide
        # by the total size of the dataset so you can
        # use a percentage below.

        if size > 1:
            size /= len(dset)
        frac_split_sizes[name] = size

    train, val, test = split_train_validation_test(
        dset,
        val_size=frac_split_sizes["val"],
        test_size=frac_split_sizes["test"],
        seed=seed)

    return train, val, test


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
                     spec_dic):

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
                                     spec_dic=spec_dic)

    elif sum(present_dic.values()) == 3:
        split_idx = three_preset_split(dset=dset, spec_dic=spec_dic)

    # extract split indices from the dictionary
    train_idx = split_idx["train"]
    val_idx = split_idx["val"]
    test_idx = split_idx["test"]

    return (train_idx, val_idx, test_idx)


def make_split(dset, job_info):

    names = ["train", "val", "test"]
    species_splits = {name: job_info[f"{name}_species"]
                      for name in names if
                      f"{name}_species" in job_info}

    # `split_sizes` is a dictionary of the form
    # {train: 0.8, val: 0.2}
    split_sizes = job_info["split_sizes"]
    seed = job_info["seed"]

    if species_splits:
        splits = split_by_species(dset=dset,
                                  species_splits=species_splits,
                                  split_sizes=split_sizes,
                                  seed=seed)
    else:
        splits = make_random_split(dset=dset,
                                   split_sizes=split_sizes,
                                   seed=seed)
    split_dic = {}
    for name, split in zip(names, splits):
        split_dic[name] = {"dset": split}

    return split_dic


def get_sampler_kwargs(this_dset, job_info):
    sampler_kwargs = {"props": this_dset.props,
                      **job_info["sampler_kwargs"]}

    ref_config_dict = job_info.get("ref_config")

    if ref_config_dict:

        ref_config_type = ref_config_dict["type"]
        ref_config_kwargs = ref_config_dict["kwargs"]
        balance_type = job_info["balance_type"]

        if balance_type == "spec_config_zhu_balance":
            ref_config_kwargs.update({"props": this_dset.props})

        print("Generating reference structures...")
        ref_dic = get_ref_dic(ref_config_type, **ref_config_kwargs)
        print("Completed generating reference structures.")

        if balance_type == "spec_config_zhu_balance":
            sampler_kwargs.update({"ref_nxyz_dic": ref_dic})
        else:
            raise NotImplementedError

    return sampler_kwargs


def split_and_sample(dset, job_info):

    print("Splitting dataset...")
    split_dic = make_split(dset=dset, job_info=job_info)

    print("Creating samplers and diabatic values for each split...")

    for key in tqdm(list(split_dic.keys())):

        sub_dic = split_dic[key]
        this_dset = sub_dic["dset"]
        sampler_kwargs = get_sampler_kwargs(this_dset=this_dset,
                                            job_info=job_info)

        sampler, cluster_rmsds = sampler_and_rmsds(
            balance_type=job_info["balance_type"],
            sampler_kwargs=sampler_kwargs)

        this_dset = add_diabat(cluster_rmsds=cluster_rmsds,
                               max_rmsd=job_info["max_diabat_rmsd"],
                               num_states=job_info["num_diabat_states"],
                               assignments=job_info["diabat_assignments"],
                               dset=this_dset,
                               diabatic_keys=job_info["diabatic_keys"])

        needs_nbrs = job_info.get("needs_nbrs", False)
        needs_angles = job_info.get("needs_angles", False)
        cutoff = job_info.get("cutoff", 5.0)

        if needs_nbrs or (needs_angles and
                          ("nbr_list" not in this_dset.props)):
            print(("Generating neighbor list with cutoff %.2f A"
                   % cutoff))
            this_dset.generate_neighbor_list(cutoff, undirected=False)

        if needs_angles:
            print("Generating angle list and directed indices")
            this_dset.generate_angle_list()
            print("Completed generating angle list.")

        split_dic[key] = {"dset": this_dset, "sampler": sampler}

    return split_dic


def save_as_chunks(split_dic, job_info):

    model_path = get_model_path(job_info)
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
            weight_path = os.path.join(direc, f"{split}_sample_dict.json")
            
            with open(weight_path, "w") as f_open:
                json.dump(weights, f_open)


def diabat_and_weights(dset, job_info):

    dset = prune_by_substruc(dset=dset,
                             substruc_smiles=job_info.get("substruc_smiles"),
                             keep_stereo=job_info.get("stereo_in_substruc",
                                                      False))
    dset.shuffle()

    split_dic = split_and_sample(dset=dset, job_info=job_info)

    print("Saving...")
    save_as_chunks(split_dic=split_dic,
                   job_info=job_info)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def flatten_dict(job_info):
    cat_info = {}
    for key, val in job_info.items():
        if isinstance(val, dict):
            cat_info.update(val)
        else:
            cat_info[key] = val
    return cat_info


def load_dset(job_info):

    if os.path.isfile(job_info["dset_path"]):
        dset_path = job_info["dset_path"]
        paths = [dset_path]

    elif os.path.isdir(job_info["dset_path"]):
        name_choices = [['dataset.pth.tar'], [
            'train.pth.tar', 'test.pth.tar', 'val.pth.tar']]
        exists = False
        for names in name_choices:
            paths = [os.path.join(job_info["dset_path"], name)
                     for name in names]
            if any([os.path.isfile(path) for path in paths]):
                exists = True
                break
        if not exists:
            msg = (f"Path {job_info['dset_path']} is neither a "
                   "directory nor file.")
            raise Exception(msg)

    print("Loading dataset...")
    for i, path in tqdm_enum(paths):
        if i == 0:
            dset = Dataset.from_file(path)
        else:
            dset += Dataset.from_file(path)
    print("Loaded!")
    return dset


def read_info_file(config_file):
    with open(config_file, "r") as f_open:
        job_info = json.load(f_open)
    if "details" in job_info:
        job_info = flatten_dict(job_info)

    for name in ALLOWED_DIVIDER_NAMES:
        if name in job_info:
            job_info.update(job_info[name])
            job_info.pop(name)

    return job_info


def parse_job_info():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help="file containing all details",
                        default="distort.json")

    args = parser.parse_args()
    config_file = args.config_file
    job_info = read_info_file(config_file)

    return job_info


def main():

    job_info = parse_job_info()
    set_seed(job_info["seed"])
    dset = load_dset(job_info)
    diabat_and_weights(dset=dset, job_info=job_info)

    print("Complete!")


if __name__ == "__main__":
    main()
