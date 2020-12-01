"""
Wrapper to get species from a summary dictionary and split them
into train/val/test using the scaffold split in ChemProp.
"""

import csv
import os
import numpy as np
import json
import shutil
import argparse
from rdkit import Chem
from tqdm import tqdm

from nff.utils import bash_command, parse_args, fprint, prop_split


def apply_transfs(props, summary_dic):
    """
    Apply transformation to quantities in the dataset. For example,
    if a requested property is log_<actual property>, then create
    this requested property by taking logs in the dataset.
    Args:
       props (list[str]): list of property names that you want to predict
       summary_dic (dict): dictionary of the form {smiles: sub_dic},
        where `sub_dic` is a dictionary with all the species properties
        apart from its conformers.
    Returns:
      None
    """

    for prop in props:
        prop_present = any([prop in sub_dic for sub_dic
                            in summary_dic.values()])
        if prop_present:
            continue

        if prop.startswith("log_"):
            base_prop = prop.split("log_")[-1]

            def transf(x): return np.log(x)

        else:
            raise Exception((f"{prop} is not in the summary "
                             "dictionary and doesn't have a prefix "
                             "corresponding to a known transformation, "
                             "such as log."))

        base_present = any([base_prop in sub_dic for sub_dic
                            in summary_dic.values()])
        if not base_present:
            raise Exception((f"{base_prop} is not in the summary "
                             "dictionary."))

        # update summary dictionary with transformed keys

        for smiles, sub_dic in summary_dic.items():
            if base_prop in sub_dic:
                sub_dic.update({prop: transf(sub_dic[base_prop])})


def to_csv(summary_dic,
           props,
           csv_file):
    """
    Write the SMILES and properties in the summary dictionary
    to a csv file.
    Args:
      summary_dic (dict): dictionary of the form {smiles: sub_dic},
        where `sub_dic` is a dictionary with all the species properties
        apart from its conformers.
      props (list[str]): list of property names that you want to predict
      csv_file (str): path to csv file that you want to write to
    Returns:
      None
    """

    columns = ['smiles'] + props
    dict_data = []
    for smiles, sub_dic in summary_dic.items():
        dic = {}
        for prop in props:
            if prop.startswith("log_"):
                base_prop = prop.split("log_")[-1]
                if base_prop in sub_dic:
                    dic[prop] = np.log(sub_dic[base_prop])

        dic = {prop: sub_dic[prop] for prop in props}
        dic["smiles"] = smiles
        dict_data.append(dic)

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)


def filter_prop_and_pickle(sample_dic, props):
    """

    Filter the SMILES strings to exclude those that don't have a known value
    of all `props`, or do not have a known path to a pickle file with conformer
    information.

    Args:
      sample_dic (dict): Sample of `summary_dic` that will be used in this dataset
      props (list[str]): list of property names that you want to predict

    Returns:
      sample_dic (dict): Updated `sample_dic` with the above filters applied.

    """

    smiles_list = [key for key, sub_dic in sample_dic.items()
                   if all([prop in sub_dic for prop in props])
                   and sub_dic.get("pickle_path") is not None]

    sample_dic = {key: sample_dic[key] for key in smiles_list}

    return sample_dic


def filter_atoms(sample_dic, max_atoms):
    """
    Filter the SMILES strings to exclude those whose atom count is above
    `max_atoms`.
    Args:
      sample_dic (dict): Sample of `summary_dic` that will be used in this dataset
      max_atoms (int): Maximum number of atoms allowed in a species
    Returns:
      sample_dic (dict): Updated `sample_dic` with the above filter applied.
    """

    # if `max_atoms` is unspecified then the default is no limit
    if max_atoms is None:
        max_atoms = float("inf")

    smiles_list = list(sample_dic.keys())
    good_smiles = []

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms <= max_atoms:
            good_smiles.append(smiles)

    sample_dic = {smiles: sample_dic[smiles] for
                  smiles in good_smiles}

    return sample_dic


def subsample(summary_dic,
              props,
              max_specs,
              max_atoms,
              dataset_type,
              seed):
    """
    Reduce the number of species according to `props`, `max_specs`,
    and `max_atoms`.

    Args:
      summary_dic (dict): dictionary of the form {smiles: sub_dic},
        where `sub_dic` is a dictionary with all the species properties
        apart from its conformers.
      props (list[str]): list of property names that you want to predict
      max_specs (int): maximum number of species allowed in dataset
      max_atoms (int): Maximum number of atoms allowed in a species
      dataset_type (str): type of problem, e.g. "classification" or 
        "regression".
      seed (int): random seed for split
    Returns:
      sample_dic (dict): Updated `sample_dic` with the above filter applied.
    """

    # filter to only include species with the requested props
    sample_dic = filter_prop_and_pickle(summary_dic, props)
    # filter to only include species with less than `max_atoms` atoms
    sample_dic = filter_atoms(sample_dic, max_atoms)

    # If you set a limit for `max_specs` and are doing classification,
    # try to keep as many of the underrepresented class as possible.
    # If you set a limit but aren't doing classification, select them
    # randomly.

    keep_smiles = prop_split(max_specs=max_specs,
                             dataset_type=dataset_type,
                             props=props,
                             sample_dic=sample_dic,
                             seed=seed)

    sample_dic = {smiles: sample_dic[smiles] for smiles in keep_smiles}

    return sample_dic


def make_split(summary_path,
               csv_folder,
               cp_folder,
               props,
               split_sizes,
               split_type,
               max_specs,
               max_atoms,
               dataset_type,
               seed):
    """
    Split the species into train, test, and validation sets.

    Args:
      summary_path (str): path to the JSON file that summarizes
        all of the information about the species, apart from their
        conformers.
      csv_folder (str): path to the folder in which we will save our
        csv files with the SMILES, properties and training splits.
      cp_folder (str): path to the ChemProp folder on your computer
      props (list[str]): list of property names that you want to predict
      split_sizes (list[float]): list of the form [train_split_size, val_split_size,
        test_split_size].
      split_type (str): how to split the data. Options can be found in the Chemprop
        script `split_data.py`. A good choice is usually `scaffold_balanced`, which splits
        in such a way that similar scaffolds are in the same split. 
      max_specs (int): maximum number of species allowed in dataset
      max_atoms (int): Maximum number of atoms allowed in a species
      dataset_type (str): type of problem, e.g. "classification" or 
        "regression".
      seed (int): random seed for split
    Returns:
      None

    """

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    # apply any transformations to the data, e.g. wanting a
    # dataset that has the log of a value instead of the
    # value itself
    apply_transfs(props, summary_dic)

    # filter based on props, max species and max number of atoms
    summary_dic = subsample(summary_dic=summary_dic,
                            props=props,
                            max_specs=max_specs,
                            max_atoms=max_atoms,
                            dataset_type=dataset_type,
                            seed=seed)

    # path csv file with SMILES and properties
    all_csv = os.path.join(csv_folder, "all.csv")
    if not os.path.isdir(csv_folder):
        os.makedirs(csv_folder)
    # write the contents of `summary_dic` to the csv
    to_csv(summary_dic, props, all_csv)

    # run the chemprop script `split_data.py` to make the splits
    # from `all.csv`

    script = os.path.join(cp_folder, "scripts", "split_data.py")
    split_str = " ".join(np.array(split_sizes).astype("str"))
    cmd = (f"source activate chemprop && "
           f"python {script} --split_type {split_type} "
           f"--split_sizes {split_str} "
           f"--data_path {all_csv} "
           f"--save_dir {csv_folder} "
           f"--seed {seed}")
    p = bash_command(cmd)
    p.wait()


def add_just_smiles(csv_folder):
    """
    Take csv files with SMILES + properties and use them to crea files
    with just the SMILES strings.
    Args:
      csv_folder (str): path to the folder in which we will save oru
        csv files with the SMILES, properties and training splits.
    Returns:
      None
    """

    for name in ['train', 'val', 'test']:
        path = os.path.join(csv_folder, name + '.csv')
        smiles_list = []

        with open(path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(readCSV):
                if i == 0:
                    continue
                smiles_list.append(row[0])

        # save to "train_smiles.csv", "val_smiles.csv", etc.
        smiles_path = os.path.join(csv_folder, f"{name}_smiles.csv")
        columns = ["smiles"]
        dict_data = [{"smiles": smiles} for smiles in smiles_list]

        with open(smiles_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)


def rename_csvs(csv_folder):
    """
    Rename the csvs saved by the chemprop split function to distinguish
    between what is just SMILES and what is SMILES + properties.
    Args:
      csv_folder (str): path to the folder in which we will save oru
        csv files with the SMILES, properties and training splits.
    Returns:
      None

    """
    for name in ['train', 'val', 'test']:
        path = os.path.join(csv_folder, name + '.csv')
        # "train_full.csv", "val_full.csv", etc.
        new_path = os.path.join(csv_folder, name + "_full.csv")
        shutil.move(path, new_path)


def summarize(csv_folder, dataset_type):
    """
    Summarize where the splits have been saved and what their contents are.
    Args:
      csv_folder (str): path to the folder in which we will save oru
        csv files with the SMILES, properties and training splits.
      dataset_type (str): type of problem, e.g. "classification" or 
        "regression".
    Returns:
      None
    """
    msgs = []
    for name in ['train', 'val', 'test', 'all']:
        if name == 'all':
            path = os.path.join(csv_folder, f"{name}.csv")
        else:
            path = os.path.join(csv_folder, f"{name}_full.csv")
        with open(path, "r") as f:
            lines = f.readlines()[1:]

        num_specs = len(lines)
        this_msg = f"{num_specs} species"
        if dataset_type == "classification":
            num_pos = len([line for line in lines
                           if int(line.split(",")[-1]) == 1])
            this_msg += f", {num_pos} positives"

        msgs.append(this_msg)

    msg = (f"Splits saved in {csv_folder}\n"
           f"Train files: train_smiles.csv and train_full.csv ({msgs[0]})\n"
           f"Validation files: val_smiles.csv and val_full.csv ({msgs[1]}) \n"
           f"Test files: test_smiles.csv and test_full.csv ({msgs[2]})\n"
           f"Combined file: all.csv ({msgs[3]})")

    fprint(msg)


def main(summary_path,
         csv_folder,
         cp_folder,
         props,
         split_sizes,
         split_type,
         max_specs,
         max_atoms,
         dataset_type,
         seed,
         **kwargs):
    """
    Split the data, write it to csvs, create new csvs with just
    SMILES and no properties, and rename the existing csvs.

    Args:
      summary_path (str): path to the JSON file that summarizes
        all of the information about the species, apart from their
        conformers.
      csv_folder (str): path to the folder in which we will save our
        csv files with the SMILES, properties and training splits.
      cp_folder (str): path to the ChemProp folder on your computer
      props (list[str]): list of property names that you want to predict
      split_sizes (list[float]): list of the form [train_split_size, val_split_size,
        test_split_size].
      split_type (str): how to split the data. Options can be found in the Chemprop
        script `split_data.py`. A good choice is usually `scaffold_balanced`, which splits
        in such a way that similar scaffolds are in the same split. 
      max_specs (int): maximum number of species allowed in dataset
      max_atoms (int): Maximum number of atoms allowed in a species
      dataset_type (str): type of problem, e.g. "classification" or 
        "regression".
      seed (int): random seed for split
    Returns:
      None
    """

    make_split(summary_path=summary_path,
               csv_folder=csv_folder,
               cp_folder=cp_folder,
               props=props,
               split_sizes=split_sizes,
               split_type=split_type,
               max_specs=max_specs,
               max_atoms=max_atoms,
               dataset_type=dataset_type,
               seed=seed)

    add_just_smiles(csv_folder)
    rename_csvs(csv_folder)

    summarize(csv_folder, dataset_type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_path', type=str,
                        help="Path to summary dictionary")
    parser.add_argument('--csv_folder', type=str,
                        help=("Name of the folder in which "
                              "you want to save the SMILES "
                              "splits"))
    parser.add_argument('--cp_folder', type=str,
                        help="Tour chemprop folder ")
    parser.add_argument('--props', type=str,
                        nargs='+',
                        help=("Name of the properties you're "
                              "predicting"))
    parser.add_argument('--split_sizes', type=float,
                        nargs="+",
                        help="Train/val/test split proportions ",
                        default=[0.8, 0.1, 0.1])
    parser.add_argument('--split_type', type=str,
                        choices=['random', 'scaffold_balanced'],
                        help=("Type of split"))
    parser.add_argument('--max_specs', type=int, default=None,
                        help=("Maximum number of species to use in your "
                              "dataset. No limit if max_specs isn't "
                              "specified."))
    parser.add_argument('--max_atoms', type=int, default=None,
                        help=("Maximum number of atoms to allow in any "
                              "species in your dataset. No limit if "
                              "max_atoms isn't specified."))
    parser.add_argument('--dataset_type', type=str,
                        choices=['regression', 'classification'],
                        help=("Type of training task."))
    parser.add_argument('--seed', type=int,
                        help=("Random seed for split"))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))
    args = parse_args(parser)

    main(**args.__dict__)
