"""
Script to run transfer learning from a 3D model using different possible metrics
and different possible chemprop options (features + MPNN, just features, or
just MPNN)
"""

import os
import argparse

from nff.utils import bash_command, parse_args, fprint

METRIC_CHOICES = ["auc",
                  "prc-auc",
                  "rmse",
                  "mae",
                  "mse",
                  "r2",
                  "accuracy",
                  "cross_entropy"]


def get_train_folder(model_folder_cp,
                     feature_folder,
                     metric,
                     feat,
                     mpnn):

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


def get_msg(feat, mpnn, metric, train_folder):

    msg = "Training a ChemProp model with "
    if feat:
        msg += "set features "
    if feat and mpnn:
        msg += "and "
    if mpnn:
        msg += "an MPNN "

    msg += f"in folder {train_folder}"

    return msg


def main(base_config_path,
         cp_folder,
         feature_folder,
         model_folder_cp,
         metrics,
         feat_options,
         mpnn_options,
         **kwargs):

    cwd = os.path.abspath(".")
    script = os.path.join(cwd, "cp_tl.py")

    for feat in feat_options:
        for mpnn in mpnn_options:
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

                msg = get_msg(feat, mpnn, metric, train_folder)
                fprint(msg)

                cmd = (f"python {script} "
                       f"--base_config_path {base_config_path} "
                       f"--metric {metric} "
                       f"--train_feat_path {train_feat_path} "
                       f"--val_feat_path {val_feat_path} "
                       f"--test_feat_path {test_feat_path} "
                       f"--train_folder {train_folder} "
                       f"--cp_folder {cp_folder} ")

                p = bash_command(cmd)
                p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config_path", type=str,
                        help=("Path to the reference config file "
                              "used to train a ChemProp model. "
                              "This file will be modified as "
                              "we loop through different arguments."))
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
                        choices=METRIC_CHOICES,
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
