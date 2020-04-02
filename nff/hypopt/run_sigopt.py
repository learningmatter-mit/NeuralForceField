import argparse
import json
from nff.hypopt.hyp_sigopt import run_loop, retrain_best
from nff.hypopt.feats import preprocess
import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('info_file',
                        type=str,
                        help=('path of the json file with information '
                              'about the hyperparameter optimization.'))
    parser.add_argument('--retrain_best',
                        action="store_true",
                        help=('find the best hyperparameters and '
                              'retrain a model with those parameters.'))
    parser.add_argument('--best_epochs',
                        type=int,
                        help=('Number of epochs to train the best model for'),
                        default=500)

    arguments = parser.parse_args()
    with open(arguments.info_file, "r") as f:
        info = json.load(f)
    # preprocess(info)

    if arguments.retrain_best:
        info.update({"num_epochs": arguments.best_epochs})
        retrain_best(**info)
    else:
        run_loop(**info)
