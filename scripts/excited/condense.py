import os
import json
import numpy as np
import shutil
import argparse

from tqdm import tqdm
from nff.data import Dataset


SPLIT_NAMES = ['train', 'val', 'test']


def aggregate(base_dir, folders):
    balances = {}
    dsets = {}

    for folder in tqdm(folders):
        for name in tqdm(SPLIT_NAMES):
            dset_path = os.path.join(base_dir, folder,
                                     f"{name}.pth.tar")
            balance_path = os.path.join(base_dir, folder,
                                        f"{name}_sample_dict.json")

            dset = Dataset.from_file(dset_path)
            with open(balance_path, 'r') as f_open:
                balance_dict = json.load(f_open)

            if name in dsets:
                dsets[name] += dset
                balances[name] = {key: balances[name][key] + balance_dict[key]
                                  for key in balance_dict.keys()}
            else:
                dsets[name] = dset
                balances[name] = balance_dict

    return balances, dsets


def get_new_groups(base_dir, num_new):
    old_folders = np.array([i for i in os.listdir(base_dir)
                            if i.isdigit() and os.path.isdir(
                                os.path.join(base_dir, i))])
    new_groups = np.array_split(old_folders, num_new)
    return new_groups


def save_to_new(new_folder,
                balances,
                dsets):

    for name in tqdm(SPLIT_NAMES):
        dset = dsets[name]
        balance_dict = balances[name]

        dset_path = os.path.join(new_folder, f"{name}.pth.tar")
        balance_path = os.path.join(new_folder, f"{name}_sample_dict.json")

        dset.save(dset_path)
        with open(balance_path, "w") as f_open:
            json.dump(balance_dict, f_open, indent=4)


def aggr_all(base_dir,
             num_new,
             new_dir,
             replace,
             **kwargs):

    new_groups = get_new_groups(base_dir=base_dir,
                                num_new=num_new)

    for i in tqdm(range(num_new)):
        folders = new_groups[i]
        balances, dsets = aggregate(base_dir=base_dir,
                                    folders=folders)
        new_folder = os.path.join(new_dir, str(i))
        if os.path.isdir(new_folder):
            if replace:
                backup_path = os.path.join(new_dir, f"{i}_backup")
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                shutil.move(new_folder, backup_path)
            else:
                msg = (f"{new_folder} exists and you asked not "
                       "to overwrite any existing folders.")
                raise Exception(msg)
        os.makedirs(new_folder)

        save_to_new(new_folder=new_folder,
                    balances=balances,
                    dsets=dsets)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help="file containing all details",
                        default="condense_config/schnet.json")

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as f_open:
        kwargs = json.load(f_open)

    return kwargs


def main():
    kwargs = parse()
    aggr_all(**kwargs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()
