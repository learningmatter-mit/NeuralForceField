import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")


from nff.hypopt.feats import preprocess
from nff.hypopt.hyp_sigopt import run_loop
import json
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('info_file',
                        type=str,
                        help=('path of the json file with information '
                              'about the hyperparameter optimization.'))

    arguments = parser.parse_args()
    with open(arguments.info_file, "r") as f:
        info = json.load(f)
    preprocess(info)
    run_loop(**info)
