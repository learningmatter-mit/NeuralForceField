#!/usr/bin/env python
import logging
import os

import torch

import nff.train.metrics
from nff.utils.script_utils import (
    get_main_parser,
    add_subparsers,
    setup_run,
    get_representation,
    get_model,
    get_trainer,
    evaluate,
    get_statistics,
    get_loaders,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_subparsers(parser)
    args = parser.parse_args()
    train_args = setup_run(args)

    # set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # define metrics
    metrics = [
        nff.train.metrics.MeanAbsoluteError(
            train_args.property, train_args.property
        ),
        nff.train.metrics.RootMeanSquaredError(
            train_args.property, train_args.property
        ),
    ]

    # splits the dataset in test, val, train sets
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=qm9, split_path=split_path, logging=logging
    )

    if args.mode == "train":

        # run training
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        trainer.train(device, n_epochs=args.n_epochs)
        logging.info("...training done!")

    elif args.mode == "eval":
        # load model
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        # run evaluation
        logging.info("evaluating...")
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... done!")
    else:
        raise NotImplementedError("Unknown mode:", args.mode)
