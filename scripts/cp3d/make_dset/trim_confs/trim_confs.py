"""
Script to copy a reference dataset into a new dataset
with fewer conformers per species.
"""


import os
import argparse
import json
import pdb
from tqdm import tqdm

from nff.data import Dataset
from nff.utils import fprint
from nff.utils.confs import trim_confs


def main(from_model_path,
         to_model_path,
         num_confs,
         conf_file,
         **kwargs):
    """
    Load the dataset, reduce the number of conformers, and save it.
    Args:
        from_model_path (str): The path to the folder in which
            the old dataset is saved.
        to_model_path (str): The path to the folder in which
            the new dataset will be saved.
        num_confs (int): Desired number of conformers per species
        conf_file (str): Path to the JSON file that tells you which
            conformer indices to use for each species.
    Returns:
        None
    """

    # load `conf_file` if given

    if conf_file is not None:
        with open(conf_file, "r") as f:
            idx_dic = json.load(f)
    else:
        idx_dic = None

    # If the folder has sub_folders 0, 1, ..., etc.,
    # then load each dataset in each sub-folder. Otherwise
    # the dataset must be in the main folder.

    folders = sorted([i for i in os.listdir(from_model_path)
                      if i.isdigit()], key=lambda x: int(x))

    if folders == []:
        folders = [""]

    # Go through each dataset, update it, and save it

    for folder in tqdm(folders):

        fprint(folder)
        for name in ["train.pth.tar", "test.pth.tar", "val.pth.tar"]:
            load_path = os.path.join(from_model_path, folder, name)
            if not os.path.isfile(load_path):
                continue
            dataset = Dataset.from_file(load_path)
            dataset = trim_confs(dataset=dataset,
                                 num_confs=num_confs,
                                 idx_dic=idx_dic)

            save_folder = os.path.join(to_model_path, folder)
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, name)
            dataset.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_model_path', type=str,
                        help="Path to model from which original data comes")
    parser.add_argument('--to_model_path', type=str,
                        help="Path to model to which new data is saved")
    parser.add_argument('--num_confs', type=int,
                        help="Number of conformers per species",
                        default=1)
    parser.add_argument('--conf_file', type=str,
                        help=("Path to json that says which conformer "
                              "to use for each species. This is optional. "
                              "If you don't specify the conformers, the "
                              "script will default to taking the `num_confs` "
                              "lowest conformers, ordered by statistical "
                              "weight."),
                        default=None)

    args = parser.parse_args()

    try:
        main(**args.__dict__)
    except Exception as e:
        fprint(e)
        pdb.post_mortem()
