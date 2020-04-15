import argparse
import json
from nff.hypopt.hyp_sigopt import run_loop, retrain_best
from nff.hypopt.feats import preprocess


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
    parser.add_argument('--make_features',
                        action='store_true',
                        help=('Make chemprop features'))
    parser.add_argument('--target_balance',
                        type=str,
                        help=('Target for balanced training'))
    parser.add_argument('--ensembles',
                        type=int,
                        help=('Number of ensembles for training'))


    arguments = parser.parse_args()
    with open(arguments.info_file, "r") as f:
        info = json.load(f)
        
    if arguments.make_features:
        preprocess(info)
    if arguments.target_balance:
        info.update({"target_balance": arguments.target_balance})
    if arguments.ensembles:
        info.update({"num_ensembles": arguments.ensembles})

    if arguments.retrain_best:
        info.update({"num_epochs": arguments.best_epochs})
        retrain_best(**info)

    else:
        run_loop(**info)
