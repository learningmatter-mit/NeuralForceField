#!/usr/bin/env python
import logging
import os

import torch

import nff.train.metrics
from nff.utils.script_utils import (
    get_main_parser,
    add_subparsers,
    setup_run,
    get_loaders,
)
from nff.train.builders import get_trainer, get_model
from nff.train.loss import build_mse_loss
from nff.train.evaluate import evaluate

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # parse arguments
    parser = get_main_parser()
    add_subparsers(parser)
    args = parser.parse_args()
    train_args = setup_run(args)

    # set device
    device = torch.device(args.device)

    # define metrics
    metrics = [
        nff.train.metrics.MeanAbsoluteError('energy'),
        nff.train.metrics.MeanAbsoluteError('force')
    ]

    model = get_model(vars(args))

    # splits the dataset in test, val, train sets
    train_loader, val_loader, test_loader = get_loaders(args, logging=logging)

    if args.mode == "train":

        # run training
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        trainer.train(device, n_epochs=args.n_epochs)
        logging.info("...training done!")

    elif args.mode == "eval":
        # load model
        model = torch.load(os.path.join(args.model_path, "best_model"))
        loss_fn = build_mse_loss(args.rho)

        # run evaluation
        logging.info("evaluating...")
        _, _, test_loss = evaluate(model, loader, loss_fn)
        logging.info("... done!")
        logging.info('loss = %.4f' % test_loss)

    else:
        raise NotImplementedError("Unknown mode:", args.mode)
