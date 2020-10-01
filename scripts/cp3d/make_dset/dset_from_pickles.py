import pickle
import random
import json
import os
import torch
import numpy as np
import argparse
import sys


from nff.data import Dataset, concatenate_dict
from nff.utils import tqdm_enum, parse_args, fprint
import copy


KEY_MAP = {"rd_mol": "nxyz",
           "boltzmannweight": "weights",
           "relativeenergy": "energy"}

# these are keys that confuse the dataset
EXCLUDE_KEYS = ["totalconfs", "datasets", "conformerweights",
                "uncleaned_smiles"]

def get_thread_dic(sample_dic, thread, num_threads):

    keys = np.array(sorted(list(
        sample_dic.keys())))
    split_keys = np.array_split(keys, num_threads)
    thread_keys = split_keys[thread]

    sample_dic = {key: sample_dic[key]
                  for key in thread_keys}

    return sample_dic


def gen_splits(sample_dic,
               pos_per_val,
               pos_per_test,
               prop,
               test_size,
               val_size):

    pos_smiles = [smiles for smiles, sub_dic
                  in sample_dic.items() if
                  sample_dic.get(prop) == 1]
    neg_smiles = [smiles for smiles, sub_dic
                  in sample_dic.items() if
                  sample_dic.get(prop) == 0]

    random.shuffle(pos_smiles)
    random.shuffle(neg_smiles)

    val_pos = pos_smiles[:pos_per_val]
    test_pos = pos_smiles[pos_per_val: pos_per_val + pos_per_test]

    neg_per_val = val_size - pos_per_val
    val_neg = neg_smiles[:neg_per_val]

    neg_per_test = test_size - pos_per_test
    test_neg = neg_smiles[neg_per_val: neg_per_val + neg_per_test]

    val_all = val_pos + val_neg
    test_all = test_pos + test_neg

    for smiles in sample_dic.keys():
        if smiles in val_all:
            sample_dic[smiles].update({"split": "val"})
        elif smiles in test_all:
            sample_dic[smiles].update({"split": "test"})
        else:
            sample_dic[smiles].update({"split": "train"})

    return sample_dic


def proportional_sample(summary_dic,
                        max_specs,
                        prop_sample_path,
                        test_size=None,
                        val_size=None,
                        pos_per_val=None,
                        pos_per_test=None,
                        prop=None,
                        thread=None,
                        num_threads=None,
                        sample_type='random'):
    """
    Sample species for a dataset so that the number of positives
    and negatives is the same proportion as in the overall
    dataset.

    """

    if os.path.isfile(prop_sample_path):
        with open(prop_sample_path, "r") as f:
            sample_dic = json.load(f)
        if thread is not None:
            sample_dic = get_thread_dic(sample_dic=sample_dic,
                                        thread=thread,
                                        num_threads=num_threads)
        return sample_dic

    if sample_type == 'random':

        smiles_list = [key for key, val in summary_dic.items()
                       if val.get("pickle_path") is not None]
        random.shuffle(smiles_list)
        if max_specs is None:
            max_specs = len(smiles_list)
        all_samples = smiles_list[:max_specs]

    elif sample_type == 'class_proportional':

        positives = []
        negatives = []

        for smiles, sub_dic in summary_dic.items():
            value = sub_dic.get(prop)
            if value is None:
                continue
            if sub_dic.get("pickle_path") is None:
                continue
            if value == 0:
                negatives.append(smiles)
            elif value == 1:
                positives.append(smiles)

        num_neg = len(negatives)
        num_pos = len(positives)

        if max_specs is None:
            max_specs = num_neg + num_pos

        # get the number of desired negatives and positives to
        # get the right proportional sampling

        num_neg_sample = int(num_neg / (num_neg + num_pos) * max_specs)
        num_pos_sample = int(num_pos / (num_neg + num_pos) * max_specs)

        # shuffle negatives and positives and extract the appropriate
        # number of each

        random.shuffle(negatives)
        random.shuffle(positives)

        neg_sample = negatives[:num_neg_sample]
        pos_sample = positives[:num_pos_sample]

        all_samples = [*neg_sample, *pos_sample]

    sample_dic = {key: summary_dic[key]
                  for key in all_samples if "pickle_path"
                  in summary_dic[key] and prop in summary_dic[key]}

    # generate train/val/test labels

    sample_dic = gen_splits(sample_dic=sample_dic,
                            pos_per_val=pos_per_val,
                            pos_per_test=pos_per_test,
                            prop=prop,
                            test_size=test_size,
                            val_size=val_size)

    with open(prop_sample_path, "w") as f:
        json.dump(sample_dic, f, indent=4, sort_keys=True)

    if thread is not None:
        sample_dic = get_thread_dic(sample_dic=sample_dic,
                                    thread=thread,
                                    num_threads=num_threads)

    return sample_dic


def load_data_from_pickle(sample_dic, max_atoms, pickle_folder):

    overall_dic = {}
    i = 0
    total_num = len(sample_dic)
    if max_atoms is None:
        max_atoms = float("inf")

    for smiles, sub_dic in sample_dic.items():

        i += 1

        pickle_path = sub_dic["pickle_path"]
        full_path = os.path.join(pickle_folder, pickle_path)
        with open(full_path, "rb") as f:
            dic = pickle.load(f)
        num_atoms = dic["conformers"][0]["rd_mol"].GetNumAtoms()
        if num_atoms > max_atoms:
            continue
        overall_dic.update({smiles: dic})

        if np.mod(i, 1000) == 0:
            fprint(("Completed loading {} of {} "
                    "dataset pickles".format(i, total_num)))

    fprint("Completed dataset pickles")

    return overall_dic


def map_key(key):
    if key in KEY_MAP:
        return KEY_MAP[key]
    else:
        return key


def fix_iters(spec_dic, actual_confs):
    new_spec_dic = {}
    for key, val in spec_dic.items():
        if key in EXCLUDE_KEYS:
            continue
        elif type(val) in [int, float, str]:
            new_spec_dic[key] = [val] * actual_confs
        else:
            new_spec_dic[key] = val

    return new_spec_dic


def get_sorted_idx(sub_dic):

    confs = sub_dic["conformers"]
    weight_list = []
    for i, conf in enumerate(confs):
        weight_list.append([i, conf["boltzmannweight"]])
    sorted_tuples = sorted(weight_list, key=lambda x: -x[-1])
    sorted_idx = [i[0] for i in sorted_tuples]

    return sorted_idx


def get_xyz(rd_mol):

    atoms = rd_mol.GetAtoms()

    atom_nums = []
    for atom in atoms:
        atom_nums.append(atom.GetAtomicNum())

    rd_conf = rd_mol.GetConformers()[0]
    positions = rd_conf.GetPositions()

    xyz = []
    for atom_num, position in zip(atom_nums, positions):
        xyz.append([atom_num, *position])

    return xyz


def renorm_weights(spec_dic):

    new_weights = np.array(spec_dic["weights"]) / sum(spec_dic["weights"])
    spec_dic["weights"] = new_weights.tolist()

    return spec_dic


def convert_data(overall_dic, max_confs):

    spec_dics = []
    if max_confs is None:
        max_confs = float("inf")

    for key, sub_dic in overall_dic.items():

        spec_dic = {map_key(key): val for key, val in sub_dic.items()
                    if key != "conformers"}

        actual_confs = min(max_confs, len(sub_dic["conformers"]))
        spec_dic = fix_iters(spec_dic, actual_confs)
        spec_dic.update({map_key(key): [] for key
                         in sub_dic["conformers"][0].keys()
                         if key not in EXCLUDE_KEYS})

        # conformers not always ordered by weight
        sorted_idx = get_sorted_idx(sub_dic)
        confs = sub_dic["conformers"]
        spec_dic["rd_mols"] = []

        for idx in sorted_idx[:actual_confs]:
            conf = confs[idx]
            for key in conf.keys():
                if key == "rd_mol":

                    nxyz = get_xyz(conf[key])
                    spec_dic["nxyz"].append(nxyz)
                    spec_dic["rd_mols"].append(conf[key])

                else:
                    new_key = map_key(key)
                    if new_key not in spec_dic:
                        continue
                    spec_dic[new_key].append(conf[key])

        spec_dic = renorm_weights(spec_dic)
        spec_dics.append(spec_dic)

    return spec_dics


def add_missing(props_list):

    key_list = [list(props.keys()) for props in props_list]
    # dictionary of the props that have each set of keys
    key_dic = {}
    for i, keys in enumerate(key_list):
        for key in keys:
            if key not in key_dic:
                key_dic[key] = []
            key_dic[key].append(i)

    # all the possible keys
    all_keys = []
    for keys in key_list:
        all_keys += keys
    all_keys = list(set(all_keys))

    # dictionary of which props dicts are missing certain keys

    missing_dic = {}
    prop_idx = list(range(len(props_list)))
    for key in all_keys:
        missing_dic[key] = [i for i in prop_idx if
                            i not in key_dic[key]]

    for key, missing_idx in missing_dic.items():
        for i in missing_idx:

            props = props_list[i]
            given_idx = key_dic[key][0]
            given_props = props_list[given_idx]
            given_val = given_props[key]

            if type(given_val) is list:
                props[key] = [None]
            elif type(given_val) is torch.Tensor:
                props[key] = torch.Tensor([np.nan])
                # in this case we need to change the
                # other props to have type float
                for good_idx in key_dic[key]:
                    other_props = props_list[good_idx]
                    other_props[key] = other_props[key].to(torch.float)
                    props_list[good_idx] = other_props

            props_list[i] = props

    return props_list


def make_nff_dataset(spec_dics, nbrlist_cutoff):

    fprint("Making dataset with %d species" % (len(spec_dics)))

    props_list = []
    nbr_list = []
    rd_mols_list = []

    for j, spec_dic in tqdm_enum(spec_dics):

        # Treat each species' data like a regular dataset
        # and use it to generate neighbor lists
        # Ignore the graph features because there's only one
        # per species right now.

        conf_keys = ["rd_mols", "bonded_nbr_list", "bond_features",
                     "atom_features"]

        # Exclude keys related to individual conformers. These
        # include conformer features, in case you've already put
        # those in your pickle files. If not we'll generate them
        # below

        small_spec_dic = {key: val for key, val in spec_dic.items()
                          if key not in conf_keys}

        dataset = Dataset(small_spec_dic, units='kcal/mol')
        mol_size = len(dataset.props["nxyz"][0])

        dataset.generate_neighbor_list(cutoff=nbrlist_cutoff)

        # now combine the neighbor lists so that this set
        # of nxyz's can be treated like one big molecule

        nbrs = dataset.props['nbr_list']
        # number of atoms in the molecule
        new_nbrs = []

        # shift by i * mol_size for each conformer
        for i in range(len(nbrs)):
            new_nbrs.append(nbrs[i] + i * mol_size)

        # add to list of conglomerated neighbor lists
        nbr_list.append(torch.cat(new_nbrs))
        dataset.props.pop('nbr_list')

        # concatenate the nxyz's
        nxyz = np.concatenate([np.array(item) for item in spec_dic["nxyz"]]
                              ).reshape(-1, 4).tolist()

        # add properties as necessary
        new_dic = {"mol_size": mol_size,
                   "nxyz": nxyz,
                   "weights": torch.Tensor(spec_dic["weights"]
                                           ).reshape(-1, 1) / sum(
                       spec_dic["weights"]),
                   "degeneracy": torch.Tensor(spec_dic["degeneracy"]
                                              ).reshape(-1, 1),
                   "energy": torch.Tensor(spec_dic["energy"]
                                          ).reshape(-1, 1),
                   "num_atoms": [len(nxyz)]}

        new_dic.update({key: val[:1] for key, val in dataset.props.items(
        ) if key not in new_dic.keys()})

        props_list.append(new_dic)
        rd_mols_list.append(spec_dic["rd_mols"])

    fprint("Finalizing...")

    # Add props that are in some datasets but not others
    props_list = add_missing(props_list)
    props_dic = concatenate_dict(*props_list)
    # make a combined dataset where the species look like they're
    # one big molecule
    big_dataset = Dataset(props_dic.copy(), units='kcal/mol')

    # give it the proper neighbor list
    big_dataset.props['nbr_list'] = nbr_list

    # generate features

    big_dataset.props["rd_mols"] = rd_mols_list
    big_dataset.featurize()

    fprint("Complete!")

    return big_dataset


def get_data_folder(dataset_folder, thread):
    if thread is None:
        return dataset_folder
    new_path = os.path.join(dataset_folder, str(thread))
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


def split_dataset(dataset, idx):
    new_dataset = copy.deepcopy(dataset)
    new_props = {}
    for key, val in dataset.props.items():
        if type(val) is list:
            new_props[key] = [val[i] for i in idx]
        else:
            new_props[key] = val[idx]
    new_dataset.props = new_props
    return new_dataset


def save_splits(dataset,
                targ_name,
                dataset_folder,
                thread,
                sample_dic):

    split_names = ["train", "val", "test"]
    split_idx = {name: [] for name in split_names}
    split_dic = {name: [] for name in split_names}

    for i, smiles in enumerate(dataset.props['smiles']):
        split_name = sample_dic[smiles]["split"]
        split_idx[split_name].append(i)

    for name in split_names:
        split_dic[name] = split_dataset(dataset, split_idx[name])

    train = split_dic["train"]
    val = split_dic["val"]
    test = split_dic["test"]

    # train, val, test = split_train_validation_test(
    #     dataset, binary=True, targ_name=targ_name)

    fprint("Saving...")
    data_folder = get_data_folder(dataset_folder, thread)
    names = ["train", "val", "test"]

    for name, dset in zip(names, [train, val, test]):
        dset_path = os.path.join(data_folder, name + ".pth.tar")
        dset.save(dset_path)


def main(max_specs,
         max_atoms,
         max_confs,
         prop,
         summary_path,
         dataset_folder,
         pickle_folder,
         prop_sample_path,
         num_threads,
         thread,
         pos_per_val,
         pos_per_test,
         test_size,
         val_size,
         sample_type,
         nbrlist_cutoff,
         **kwargs):

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    fprint("Generating proportional sample...")

    sample_dic = proportional_sample(summary_dic=summary_dic,
                                     prop=prop,
                                     max_specs=max_specs,
                                     prop_sample_path=prop_sample_path,
                                     thread=thread,
                                     num_threads=num_threads,
                                     pos_per_val=pos_per_val,
                                     pos_per_test=pos_per_test,
                                     test_size=test_size,
                                     val_size=val_size,
                                     sample_type=sample_type)

    fprint("Loading data from pickle files...")
    overall_dic = load_data_from_pickle(sample_dic, max_atoms, pickle_folder)

    fprint("Converting data...")
    spec_dics = convert_data(overall_dic, max_confs)

    fprint("Combining to make NFF dataset...")
    dataset = make_nff_dataset(spec_dics=spec_dics,
                               nbrlist_cutoff=nbrlist_cutoff)
    fprint("Creating test/train/val splits...")
    save_splits(dataset=dataset,
                targ_name=prop,
                dataset_folder=dataset_folder,
                thread=thread,
                sample_dic=sample_dic)

    fprint("Complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_specs', type=int, default=None,
                        help=("Maximum number of species to use in your "
                              "dataset. No limit if max_specs isn't "
                              "specified."))

    parser.add_argument('--max_atoms', type=int, default=None,
                        help=("Maximum number of atoms to allow in any "
                              "species in your dataset. No limit if "
                              "max_atoms isn't specified."))
    parser.add_argument('--max_confs', type=int, default=None,
                        help=("Maximum number of conformers to allow in any "
                              "species in your dataset. No limit if "
                              "max_confs isn't specified."))

    parser.add_argument('--nbrlist_cutoff', type=float, default=5,
                        help=("Cutoff for 3D neighbor list"))

    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--pickle_folder', type=str)
    parser.add_argument('--prop_sample_path', type=str)
    parser.add_argument('--num_threads', type=int, default=None)
    parser.add_argument('--thread', type=int, default=None)

    parser.add_argument('--sample_type', type=str, default='random',
                        choices=['random', 'class_proportional'],
                        help=("Strategy of sampling species to make dataset. "
                              "Current options are random and "
                              "class_proportional. `random` samples "
                              "randomly, and `class_proportional` generates "
                              "a new dataset with the same proportion of "
                              "positive and negative classes as in the "
                              "set we're sampling form."))
    parser.add_argument('--prop', type=str, default=None,
                        help=("Name of property for which to generate "
                              "a proportional classification sample"))
    parser.add_argument('--pos_per_val', type=int, default=None)
    parser.add_argument('--pos_per_test', type=int, default=None)

    parser.add_argument('--test_size', type=int, default=None,
                        help=("Absolute size of test set (number of species) "))
    parser.add_argument('--val_size', type=int, default=None,
                        help=("Absolute size of validation set (number of species) "))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
