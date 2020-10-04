"""Helper function to setup the run from the command line.

Adapted from https://github.com/atomistic-machine-learning/schnetpack/blob/dev/src/schnetpack/utils/script_utils/setup.py
"""
import os
import logging
from shutil import rmtree

from nff.utils.tools import to_json, set_random_seed, read_from_json

__all__ = ["setup_run"]


def setup_run(args):
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.model_path, "args.json")

    # absolute paths
    argparse_dict['data_path'] = os.path.abspath(argparse_dict['data_path'])
    argparse_dict['model_path'] = os.path.abspath(argparse_dict['model_path'])

    if args.mode == "train":
        if args.overwrite and os.path.exists(args.model_path):
            logging.info("existing model will be overwritten...")
            rmtree(args.model_path)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        to_json(jsonpath, argparse_dict)

        set_random_seed(args.seed)
        train_args = args

    else:
        train_args = read_from_json(jsonpath)

    return train_args
