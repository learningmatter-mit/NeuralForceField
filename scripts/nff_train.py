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
        nff.train.metrics.MeanAbsoluteError('energy_grad')
    ]

    model = get_model(vars(args))

    if args.mode == "train":

        # splits the dataset in test, val, train sets
        train_loader, val_loader, test_loader = get_loaders(args, logging=logging)

        # run training
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        trainer.train(device, n_epochs=args.n_epochs)
        logging.info("...training done!")

    elif args.mode == "eval":
        # load model
        model = torch.load(os.path.join(args.model_path, "best_model"))
        loss_fn = build_mse_loss(args)
        test_loader = get_loaders(args, logging=logging)

        # run evaluation
        logging.info("evaluating...")
        _, _, test_loss = evaluate(model, test_loader, loss_fn, args.device)
        logging.info('loss = %.4f' % test_loss)
        logging.info("... done!")

    else:
        raise NotImplementedError("Unknown mode:", args.mode)
