import sys
import os

this_path = os.path.abspath(os.path.dirname(__file__))
nff_dir = "/".join(this_path.split("/")[:-2])
sys.path.insert(0, nff_dir)

import pdb
pdb.set_trace()


import msgpack
import os
import argparse

from nff.data.features.graph import (
    single_feats_from_dic, add_single_feats_to_dataset)
from nff.data import Dataset

NOBACKUP = "/nobackup1/saxelrod/data"
MSG_FILE = os.path.join(NOBACKUP, "charge_cleandrugs_post_process.msgpack")
SINGLE_FEAT_FILE = os.path.join(NOBACKUP, "drugs_single_feats.msgpack")
DATASET_FILE = os.path.join(NOBACKUP, "drugs_dataset_100_confs.pth.tar")


def save_single_feats(msg_file, save_file, remove_old):

    unpacker = msgpack.Unpacker(open(msg_file, "rb"))
    single_feat_dic = {}

    if os.path.isfile(save_file) and remove_old:
        os.remove(save_file)

    for i, dic in enumerate(unpacker):

        print(("Converting features in dictionary "
               "{} to single, binary features.".format(
                   i)))

        single_feat_dic = single_feats_from_dic(dic)
        mp_form = msgpack.packb(single_feat_dic, use_bin_type=True)

        with open(save_file, "ab") as f:
            f.write(mp_form)


def update_dataset(dataset_file, single_feat_file):

    print("Loading dataset...")
    dataset = Dataset.from_file(dataset_file)
    print("Adding features...")
    dataset = add_single_feats_to_dataset(dataset=dataset,
                                          single_feat_file=single_feat_file)
    print("Saving dataset...")
    dataset.save(dataset_file)


def main(msg_file,
         single_feat_file,
         dataset_file,
         remove_old,
         save_singles,
         update_dataset):

    if save_singles:
        print(("Extracting graph features that "
               "are the same for all conformers."))
        save_single_feats(msg_file=msg_file,
                          save_file=single_feat_file,
                          remove_old=remove_old)
    if update_dataset:
        print("Updating dataset with features.")
        update_dataset(dataset_file=dataset_file,
                       single_feat_file=single_feat_file)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msg_file', type=str, default=MSG_FILE,
                        help=('Name of '
                              'messagepack file with features'))
    parser.add_argument('--single_feat_file',
                        type=str, default=SINGLE_FEAT_FILE,
                        help=('Name of messagepack file that you '
                              'want to save the single features to.'))
    parser.add_argument('--dataset_file', type=str, default=DATASET_FILE,
                        help=('Name of '
                              'the dataset file you want to load.'))
    parser.add_argument('--keep_old', action='store_true',
                        help=('Keep old single features file.'))
    parser.add_argument('--save_singles', action='store_true',
                        help=('Save new single features'))
    parser.add_argument('--update_dataset', action='store_true',
                        help=('Update dataset with new features'))

    arguments = parser.parse_args()

    main(msg_file=arguments.msg_file,
         single_feat_file=arguments.single_feat_file,
         dataset_file=arguments.dataset_file,
         remove_old=(not arguments.keep_old),
         save_singles=arguments.save_singles,
         update_dataset=arguments.update_dataset)
