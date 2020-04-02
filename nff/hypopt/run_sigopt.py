import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")


from nff.hypopt.feats import preprocess
from nff.hypopt.hyp_sigopt import run_loop, retrain_best
import json
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('info_file',
                        type=str,
                        help=('path of the json file with information '
                              'about the hyperparameter optimization.'))
    parser.add_argument('--retrain_best',
                        action="store_true",
                        help=('find the best model and train it.'))
    parser.add_argument('--best_epochs',
                        type=int,
                        help=('Number of epochs to train the best model for'),
                        default=500)


    arguments = parser.parse_args()
    with open(arguments.info_file, "r") as f:
        info = json.load(f)
    preprocess(info)

    if arguments.retrain_best:
    	info.update({"num_epochs": arguments.best_epochs})
    	retrain_best(**info)
    else:
	    run_loop(**info)
