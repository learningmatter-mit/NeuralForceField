"""
Script to run transfer learning from a 3D model using different possible metrics
and different possible chemprop options (features + MPNN, just features, or
just MPNN)
"""

import os
import argparse

from nff.utils import (bash_command, parse_args, fprint,
                       CHEMPROP_METRICS)


def get_train_folder(model_folder_cp,
                     feature_folder,
                     metric,
                     feat,
                     mpnn):
    """
    Create a name for the training folder based on what kind of model
    you're making.
    Args:
      model_folder_cp (str): directory in which you'll be saving your model
        folders
      feature_folder (str): directory with files for the features of the species
      feat (bool): whether this model is being trained with external features
      mpnn (bool): whether this model is being trained with an mpnn (vs. just with
        external features)
    Returns:
      folder (str): path to the training folder
    """

    model_name_3d = feature_folder.split("/")[-1]
    if model_name_3d == "":
        model_name_3d = feature_folder.split("/")[-2]

    name = f"{model_name_3d}"
    if feat and not mpnn:
        name += "_feats_no_mpnn"
    elif feat and mpnn:
        name += "_feats_mpnn"
    elif mpnn:
        name += "_just_mpnn"
    name += f"_from_{metric}"

    folder = os.path.join(model_folder_cp, name)

    return folder


def get_msg(feat, mpnn, train_folder):
    """
    Create a message telling the user what kind of model we're training.
    Args:
      feat (bool): whether this model is being trained with external features
      mpnn (bool): whether this model is being trained with an mpnn (vs. just with
        external features)
      train_folder (str): path to the training folder
    Returns:
      msg (str): the message
    """

    msg = "Training a ChemProp model with "
    if feat:
        msg += "set features "
    if feat and mpnn:
        msg += "and "
    if mpnn:
        msg += "an MPNN "

    msg += f"in folder {train_folder}\n"

    return msg


def main(base_config_path,
         hyp_config_path,
         use_hyperopt,
         rerun_hyperopt,
         cp_folder,
         feature_folder,
         model_folder_cp,
         metrics,
         feat_options,
         mpnn_options,
         **kwargs):
    """
    Run transfer learning using fingerprints from 3D models evaluated by performance
    on a variety of metrics. Different models are trained with the fingerprints and
    with or without an MPNN.
    Args:
      base_config_path (str): where your basic job config file
        is, with parameters that may or may not be changed depending
        on the given run
      hyp_config_path (str): where your basic hyperopt job config file
        is, with parameters that may or may not be changed depending
        on the given run
      use_hyperopt (bool): do a hyperparameter optimization before training
        the model
      rerun_hyperopt (bool): whether to rerun hyperparameter optimization if
        `hyp_folder` already exists and has the completion file
        `best_params.json`.
      cp_folder (str): path to the chemprop folder on your computer
      feature_folder (str): directory with files for the features of the species
      model_folder_cp (str): directory in which you'll be saving your model
        folders
      metrics (list[str]): metrics you want to use
      feat_options (list[bool]): options you want to use for features. For example,
        [True, False] means you want to train one model with features and one without,
        while [True] just means you want to train one with features.
      mpnn_options (list[bool]): same idea as `feat_options`, but for whether or not to
        use an MPNN
    Returns:
      None
    """

    cwd = os.path.abspath(".")
    script = os.path.join(cwd, "cp_tl.py")

    for feat in feat_options:
        for mpnn in mpnn_options:
            # can't run anything without either features or an MPNN
            if (not feat) and (not mpnn):
                continue
            for metric in metrics:

                paths = []
                for split in ['train', 'val', 'test']:
                    paths.append(os.path.join(feature_folder,
                                              f"{split}_{metric}.npz"))

                train_feat_path, val_feat_path, test_feat_path = paths

                train_folder = get_train_folder(
                    model_folder_cp=model_folder_cp,
                    feature_folder=feature_folder,
                    metric=metric,
                    feat=feat,
                    mpnn=mpnn)

                msg = get_msg(feat, mpnn, train_folder)
                fprint(msg)

                cmd = (f"python {script} "
                       f"--base_config_path {base_config_path} "
                       f"--hyp_config_path {hyp_config_path} "
                       f"--metric {metric} "
                       f"--train_feat_path {train_feat_path} "
                       f"--val_feat_path {val_feat_path} "
                       f"--test_feat_path {test_feat_path} "
                       f"--train_folder {train_folder} "
                       f"--cp_folder {cp_folder} ")

                if use_hyperopt:
                    cmd += "--use_hyperopt "
                if rerun_hyperopt:
                    cmd += "--rerun_hyperopt "
                if not mpnn:
                    cmd += "--features_only "
                if not feat:
                    cmd += "--no_features "

                p = bash_command(cmd)
                p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config_path", type=str,
                        help=("Path to the reference config file "
                              "used to train a ChemProp model. "
                              "This file will be modified as "
                              "we loop through different arguments."))
    parser.add_argument("--hyp_config_path", type=str, default=None,
                        help=("Same as `base_config_path`, but "
                              "for the hyperparameter optimization "
                              "stage"))
    parser.add_argument("--use_hyperopt", action='store_true',
                        help=("Do hyperparameter optimization before "
                              "training "))
    parser.add_argument("--rerun_hyperopt", action='store_true',
                        help=("Rerun hyperparameter optimization even if "
                              "it has been done already. "))

    parser.add_argument("--cp_folder", type=str,
                        help=("Path to ChemProp folder."))
    parser.add_argument("--feature_folder", type=str,
                        help=("Folder where features are stored."))
    parser.add_argument("--model_folder_cp", type=str,
                        help=("Folder in which you will train your "
                              "ChemProp model. Models with different "
                              "parameters will get their own folders, "
                              "each located in `model_folder_cp`."))
    parser.add_argument("--metrics", type=str,
                        nargs='+',
                        choices=CHEMPROP_METRICS,
                        help=("Metrics for which to evaluate "
                              "the model performance. You can "
                              "choose as many as you want; "
                              "different models will be trained "
                              "using the different metrics."))

    parser.add_argument("--feat_options", type=bool,
                        nargs="+",
                        help=("Whether to add 3D features to "
                              "ChemProp model. You can either specify "
                              "True, False or both True and False. "
                              "If you specify both then models both with "
                              "and without the features will be trained, "
                              "allowing you to compare their performance."))

    parser.add_argument("--mpnn_options", type=bool,
                        nargs="+",
                        help=("Whether to add an MPNN to the "
                              "ChemProp model or just use the "
                              "3D features. You can either specify "
                              "True, False or both True and False. "
                              "If you specify both then models both with "
                              "and without an MPNN will be trained, "
                              "allowing you to compare their performance."))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
