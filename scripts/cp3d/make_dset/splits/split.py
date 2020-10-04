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
import random
import copy
from rdkit import Chem
from tqdm import tqdm

from nff.utils import bash_command, parse_args, fprint


def to_csv(summary_dic,
           props,
           csv_file):

    columns = ['smiles'] + props
    dict_data = []
    for smiles, sub_dic in summary_dic.items():
        dic = {prop: sub_dic[prop] for prop in props}
        dic["smiles"] = smiles
        dict_data.append(dic)

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)


def filter_prop_and_pickle(sample_dic, props):

    # filter for both having the prop and having an associated
    # pickle file

    smiles_list = [key for key, sub_dic in sample_dic.items()
                   if all([prop in sub_dic for prop in props])
                   and sub_dic.get("pickle_path") is not None]

    sample_dic = {key: sample_dic[key] for key in smiles_list}

    return sample_dic


def filter_atoms(sample_dic, max_atoms):

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
              dataset_type):

    sample_dic = copy.deepcopy(summary_dic)
    sample_dic = filter_prop_and_pickle(sample_dic, props)
    sample_dic = filter_atoms(sample_dic, max_atoms)

    if max_specs is not None and dataset_type == "classification":

        msg = "Not implemented for multiclass"
        assert len(props) == 1, msg

        prop = props[0]
        pos_smiles = [key for key, sub_dic in sample_dic.items()
                      if sub_dic.get(prop) == 1]
        neg_smiles = [key for key, sub_dic in sample_dic.items()
                      if sub_dic.get(prop) == 0]

        if len(pos_smiles) < len(neg_smiles):
            underrep = pos_smiles
            overrep = neg_smiles
        else:
            underrep = neg_smiles
            overrep = pos_smiles

        if max_specs >= 2 * len(underrep):
            # keep all of the underrepresented class
            random.shuffle(overrep)
            num_left = max_specs - len(underrep)
            keep_smiles = underrep + overrep[:num_left]

        else:
            random.shuffle(underrep)
            random.shuffle(overrep)
            keep_smiles = (underrep[:max_specs // 2]
                           + overrep[max_specs // 2:])
    else:

        keep_smiles = list(sample_dic.keys())
        random.shuffle(keep_smiles)

    if max_specs is not None:
        keep_smiles = keep_smiles[:max_specs]

    sample_dic = {smiles: summary_dic[smiles] for smiles in keep_smiles}

    return sample_dic


def make_split(summary_path,
               csv_folder,
               cp_folder,
               props,
               split_sizes,
               split_type,
               max_specs,
               max_atoms,
               dataset_type):

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    # filter based on max species and max number of atoms
    summary_dic = subsample(summary_dic=summary_dic,
                            props=props,
                            max_specs=max_specs,
                            max_atoms=max_atoms,
                            dataset_type=dataset_type)

    all_csv = os.path.join(csv_folder, "all.csv")
    to_csv(summary_dic, props, all_csv)

    script = os.path.join(cp_folder, "scripts", "split_data.py")
    split_str = " ".join(np.array(split_sizes).astype("str"))
    cmd = (f"python {script} --split_type {split_type} "
           f"--split_sizes {split_str} "
           f"--data_path {all_csv} "
           f"--save_dir {csv_folder}")

    p = bash_command(cmd)
    p.wait()


def add_just_smiles(csv_folder):

    for name in ['train', 'val', 'test']:
        path = os.path.join(csv_folder, name + '.csv')
        smiles_list = []

        with open(path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(readCSV):
                if i == 0:
                    continue
                smiles_list.append(row[0])

        smiles_path = os.path.join(csv_folder, f"{name}_smiles.csv")
        columns = ["smiles"]
        dict_data = [{"smiles": smiles} for smiles in smiles_list]

        with open(smiles_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)


def rename_csvs(csv_folder):
    for name in ['train', 'val', 'test']:
        path = os.path.join(csv_folder, name + '.csv')
        new_path = os.path.join(csv_folder, name + "_full.csv")
        shutil.move(path, new_path)


def summarize(csv_folder, dataset_type):
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
         **kwargs):

    make_split(summary_path=summary_path,
               csv_folder=csv_folder,
               cp_folder=cp_folder,
               props=props,
               split_sizes=split_sizes,
               split_type=split_type,
               max_specs=max_specs,
               max_atoms=max_atoms,
               dataset_type=dataset_type)

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
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))
    args = parse_args(parser)

    main(**args.__dict__)
