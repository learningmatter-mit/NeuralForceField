import sys
# # sys.path.insert(0, "/home/saxelrod/repo/nff/covid/NeuralForceField")
# sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")
sys.path.insert(0, "/home/saxelrod/repo/nff/covid/NeuralForceField")

from nff.data import Dataset, concatenate_dict, split_train_validation_test
import pdb
import argparse
import numpy as np
import torch
import os
import json
import random
import pickle


# RDKIT_FOLDER = "/home/saxelrod/Repo/projects/geom/tutorials/rdkit_folder"
RDKIT_FOLDER = "/home/saxelrod/fock/Repo/projects/geom/tutorials/rdkit_folder"

PICKLE_FOLDER = os.path.join(RDKIT_FOLDER, "drugs")
SUMMARY_PATH = os.path.join(RDKIT_FOLDER, "summary_drugs.json")
# DATASET_PATH = "/home/saxelrod/final_covid_train/d_for_sigopt.pth.tar"
DATASET_PATH = "/nobackup1/saxelrod/models/1057"

FEAT_DIC_PATH = ("/home/saxelrod/fock/Repo/projects/geom/tutorials"
                 "/features/file_dic.json")
PROP_SAMPLE_PATH = "/home/saxelrod/fock/final_covid_train/prop_sample.json"


PROP = "sars_cov_one_cl_protease_active"
NUM_SPECS = 25000
# NUM_SPECS = 300

KEY_MAP = {"rd_mol": "nxyz", "boltzmannweight": "weights",
           "relativeenergy": "energy"}

EXCLUDE_KEYS = ["totalconfs", "datasets"]
FEATURE_KEYS = ["bonded_nbr_list", "atom_features", "bond_features"]
MAX_ATOMS = 100
NUM_CONFS = 100


def get_thread_dic(sample_dic, thread, num_threads):

    keys = np.array(sorted(list(
        sample_dic.keys())))
    split_keys = np.array_split(keys, num_threads)
    thread_keys = split_keys[thread]
    sample_dic = {key: sample_dic[key]
                  for key in thread_keys}

    return sample_dic


def proportional_sample(summary_dic,
                        prop,
                        num_specs,
                        prop_sample_path,
                        thread=None,
                        num_threads=None):
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

    # get the number of desired negatives and positives to
    # get the right proportional sampling

    num_neg_sample = int(num_neg / (num_neg + num_pos) * num_specs)
    num_pos_sample = int(num_pos / (num_neg + num_pos) * num_specs)

    # shuffle negatives and positives and extract the appropriate
    # number of each

    random.shuffle(negatives)
    random.shuffle(positives)

    neg_sample = negatives[:num_neg_sample]
    pos_sample = positives[:num_pos_sample]

    all_samples = [*neg_sample, *pos_sample]
    sample_dic = {key: summary_dic[key]["pickle_path"]
                  for key in all_samples if "pickle_path"
                  in summary_dic[key]}

    with open(prop_sample_path, "w") as f:
        json.dump(sample_dic, f, indent=4, sort_keys=True)

    if thread is not None:
        sample_dic = get_thread_dic(sample_dic=sample_dic,
                                    thread=thread,
                                    num_threads=num_threads)

    return sample_dic


def load_data_from_pickle(sample_dic, max_atoms, rdkit_folder):

    overall_dic = {}
    i = 0
    total_num = len(sample_dic)

    for smiles, pickle_path in sample_dic.items():

        i += 1

        full_path = os.path.join(rdkit_folder, pickle_path)
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


def convert_data(overall_dic, num_confs, feature_dic):

    fprint("Adding features...")
    overall_dic = add_features(overall_dic, feature_dic)
    fprint("Finished adding features")
    spec_dics = []

    for key, sub_dic in overall_dic.items():

        spec_dic = {map_key(key): val for key, val in sub_dic.items()
                    if key != "conformers"}

        actual_confs = min(num_confs, len(sub_dic["conformers"]))
        spec_dic = fix_iters(spec_dic, actual_confs)
        spec_dic.update({map_key(key): [] for key
                         in sub_dic["conformers"][0].keys()})

        # conformers not always ordered by weight
        sorted_idx = get_sorted_idx(sub_dic)
        confs = sub_dic["conformers"]

        for idx in sorted_idx[:num_confs]:
            conf = confs[idx]
            for key in conf.keys():
                if key == "rd_mol":
                    nxyz = get_xyz(conf[key])
                    spec_dic["nxyz"].append(nxyz)
                else:
                    spec_dic[map_key(key)].append(conf[key])

        spec_dic = renorm_weights(spec_dic)
        spec_dic = duplicate_features(spec_dic)
        spec_dics.append(spec_dic)

    return spec_dics


def add_features(overall_dic, feature_path_dic):

    bad_keys = []
    i = 0
    total_num = len(overall_dic)

    for key, sub_dic in overall_dic.items():

        feature_path = feature_path_dic.get(key)
        if feature_path is None:
            bad_keys.append(key)
            continue
        if feature_path.startswith("/home/saxelrod/Repo"):
            feature_path = feature_path.replace("Repo",
                                                "fock/Repo")
        with open(feature_path, "rb") as f:
            feature_dic = pickle.load(f)
        overall_dic[key].update(feature_dic)

        if np.mod(i, 1000) == 0:
            fprint(("Completed loading {} of {} "
                   "dataset feature pickles".format(i, total_num)))

        i += 1

    fprint("Completed loading features")

    for key in bad_keys:
        overall_dic.pop(key)

    return overall_dic


def duplicate_features(spec_dic):
    """
    Only applies when the bond and atom features
    are all the same for every conformer.
    """

    num_confs = len(spec_dic['weights'])
    for key in ["atom_features", "bond_features"]:
        spec_dic[key] = torch.cat([spec_dic[key]] * num_confs)

    nbrs = [spec_dic['bonded_nbr_list']] * num_confs
    mol_size = len(spec_dic["nxyz"][0])

    # number of atoms in the molecule
    new_nbrs = []

    # shift by i * mol_size for each conformer
    for i in range(len(nbrs)):
        new_nbrs.append(nbrs[i] + i * mol_size)

    bonded_nbr_list = torch.cat(new_nbrs)
    spec_dic["bonded_nbr_list"] = bonded_nbr_list

    return spec_dic

def fprint(msg):
    print(msg)
    sys.stdout.flush()

def make_nff_dataset(spec_dics, gen_nbrs=True, nbrlist_cutoff=5.0):

    fprint("Making dataset with %d species" % (len(spec_dics)))

    props_list = []
    nbr_list = []

    for j, spec_dic in enumerate(spec_dics):

        # Treat each species' data like a regular dataset
        # and use it to generate neighbor lists
        # Ignore the graph features because there's only one
        # per species right now.

        small_spec_dic = {key: val for key, val in spec_dic.items()
                          if key not in FEATURE_KEYS}

        dataset = Dataset(small_spec_dic, units='kcal/mol')
        mol_size = len(dataset.props["nxyz"][0])

        if gen_nbrs:
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

        # add back the features

        new_dic.update({key: [spec_dic[key]] for key in FEATURE_KEYS})
        props_list.append(new_dic)

        fprint("{} of {} complete".format(j + 1, len(spec_dics)))

    fprint("Finalizing...")
    props_dic = concatenate_dict(*props_list)
    # make a combined dataset where the species look like they're
    # one big molecule
    big_dataset = Dataset(props_dic.copy(), units='kcal/mol')

    # give it the proper neighbor list
    if gen_nbrs:
        big_dataset.props['nbr_list'] = nbr_list

    fprint("Complete!")

    return big_dataset


def get_data_folder(dataset_path, thread):
    if thread is None:
        return dataset_path
    new_path = os.path.join(dataset_path, str(thread))
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


def save_splits(dataset,
                targ_name,
                dataset_path,
                thread):

    train, val, test = split_train_validation_test(
        dataset, binary=True, targ_name=targ_name)

    fprint("Saving...")
    data_folder = get_data_folder(dataset_path, thread)
    names = ["train", "val", "test"]

    for name, dset in zip(names, [train, val, test]):
        dset_path = os.path.join(data_folder, name + ".pth.tar")
        dset.save(dset_path)


def main(num_specs,
         max_atoms,
         num_confs,
         prop,
         summary_path,
         feat_dic_path,
         dataset_path,
         rdkit_folder,
         prop_sample_path,
         only_samples,
         num_threads,
         thread):

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    with open(feat_dic_path, "r") as f:
        feature_path_dic = json.load(f)

    fprint("Generating proportional sample...")
    sample_dic = proportional_sample(summary_dic=summary_dic,
                                     prop=prop,
                                     num_specs=num_specs,
                                     prop_sample_path=prop_sample_path,
                                     thread=thread,
                                     num_threads=num_threads)
    if only_samples:
        return

    fprint("Loading data from pickle files...")
    overall_dic = load_data_from_pickle(sample_dic, max_atoms, rdkit_folder)

    fprint("Converting data...")
    spec_dics = convert_data(overall_dic, num_confs, feature_path_dic)

    fprint("Combining to make NFF dataset...")
    dataset = make_nff_dataset(spec_dics=spec_dics,
                               gen_nbrs=True,
                               nbrlist_cutoff=5.0)
    fprint("Creating test/train/val splits...")
    save_splits(dataset=dataset,
                targ_name=prop,
                dataset_path=dataset_path,
                thread=thread)

    fprint("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_specs', type=int, default=NUM_SPECS)
    parser.add_argument('--max_atoms', type=int, default=MAX_ATOMS)
    parser.add_argument('--num_confs', type=int, default=NUM_CONFS)
    parser.add_argument('--prop', type=str, default=PROP)
    parser.add_argument('--summary_path', type=str, default=SUMMARY_PATH)
    parser.add_argument('--feat_dic_path', type=str, default=FEAT_DIC_PATH)
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
    parser.add_argument('--rdkit_folder', type=str, default=RDKIT_FOLDER)
    parser.add_argument('--prop_sample_path', type=str,
                        default=PROP_SAMPLE_PATH)
    parser.add_argument('--only_samples', action='store_true')
    parser.add_argument('--num_threads', type=int, default=None)
    parser.add_argument('--thread', type=int, default=None)

    arguments = parser.parse_args()

    try:
        main(**arguments.__dict__)
    except Exception as e:
        fprint(e)
        pdb.post_mortem()
