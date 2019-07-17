"""Argument parsing from the command line.
From https://github.com/atomistic-machine-learning/schnetpack/blob/dev/src/schnetpack/utils/script_utils/script_parsing.py
"""

import argparse


def get_main_parser():
    """ Setup parser for command line arguments """
    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument(
        "--cuda", help="Set flag to use GPU(s)", action="store_true"
    )
    cmd_parser.add_argument(
        "--parallel",
        help="Run data-parallel on all available GPUs (specify with environment"
        " variable CUDA_VISIBLE_DEVICES)",
        action="store_true",
    )
    cmd_parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for training and prediction (default: %(default)s)",
        default=100,
    )
    return cmd_parser


def add_subparsers(cmd_parser, defaults={}):
    ## training
    train_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    train_parser.add_argument("modelpath", help="Destination for models and logs")
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Set random seed for torch and numpy."
    )
    train_parser.add_argument(
        "--overwrite", help="Remove previous model directory.", action="store_true"
    )

    train_parser.add_argument(
        "--split",
        help="Split into [train] [validation] and use remaining for testing",
        type=int,
        nargs=2,
        default=[None, None],
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate (default: %(default)s)",
        default=1e-4,
    )
    train_parser.add_argument(
        "--lr_patience",
        type=int,
        help="Epochs without improvement before reducing the learning rate "
        "(default: %(default)s)",
        default=25 if "lr_patience" not in defaults.keys() else defaults["lr_patience"],
    )
    train_parser.add_argument(
        "--lr_decay",
        type=float,
        help="Learning rate decay (default: %(default)s)",
        default=0.5,
    )
    train_parser.add_argument(
        "--lr_min",
        type=float,
        help="Minimal learning rate (default: %(default)s)",
        default=1e-6,
    )

    train_parser.add_argument(
        "--logger",
        help="Choose logger for training process (default: %(default)s)",
        choices=["csv", "tensorboard"],
        default="csv",
    )
    train_parser.add_argument(
        "--log_every_n_epochs",
        type=int,
        help="Log metrics every given number of epochs (default: %(default)s)",
        default=1,
    )
    train_parser.add_argument(
        "--n_epochs",
        help="Maximum number of training epochs (default: %(default)s)",
        default=1000,
    )
    train_parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of training epochs (default: %(default)s)",
        default=5000,
    )

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument("modelpath", help="Path of stored model")
    eval_parser.add_argument(
        "--split",
        help="Evaluate trained model on given split",
        choices=["train", "validation", "test"],
        default=["test"],
        nargs="+",
    )

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="sum"
        if "aggragation_mode" not in defaults.keys()
        else defaults["aggragation_mode"],
        choices=["sum", "avg"],
        help=" (default: %(default)s)",
    )

    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument(
        "--features",
        type=int,
        help="Size of atom-wise representation",
        default=256 if "features" not in defaults.keys() else defaults["features"],
    )
    schnet_parser.add_argument(
        "--n_filters", type=int, help="Size of atom-wise representation", default=25
    )
    schnet_parser.add_argument(
        "--interactions", type=int, help="Number of interaction blocks", default=6
    )
    schnet_parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Cutoff radius of local environment (default: %(default)s)",
    )

    schnet_parser.add_argument(
        "--cutoff_function",
        help="Functional form of the cutoff",
        choices=["hard", "cosine", "mollifier"],
        default="hard",
    )
    schnet_parser.add_argument(
        "--num_gaussians",
        type=int,
        default=25,
        help="Number of Gaussians to expand distances (default: %(default)s)",
    )

    ## setup subparser structure
    cmd_subparsers = cmd_parser.add_subparsers(
        dest="mode", help="Command-specific arguments"
    )
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser("train", help="Training help")
    subparser_eval = cmd_subparsers.add_parser("eval", help="Eval help")

    subparser_export = cmd_subparsers.add_parser("export", help="Export help")
    subparser_export.add_argument("modelpath", help="Path of stored model")
    subparser_export.add_argument(
        "destpath", help="Destination path for exported model"
    )

    train_subparsers = subparser_train.add_subparsers(
        dest="model", help="Model-specific arguments"
    )
    train_subparsers.required = True
    train_subparsers.add_parser(
        "schnet", help="SchNet help", parents=[train_parser, schnet_parser]
    )

    eval_subparsers = subparser_eval.add_subparsers(
        dest="model", help="Model-specific arguments"
    )
    eval_subparsers.required = True
    eval_subparsers.add_parser(
        "schnet", help="SchNet help", parents=[eval_parser, schnet_parser]
    )
