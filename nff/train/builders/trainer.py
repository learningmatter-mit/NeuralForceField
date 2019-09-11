"""Helper function to create a trainer for a given model.

Adapted from https://github.com/atomistic-machine-learning/schnetpack/blob/dev/src/schnetpack/utils/script_utils/training.py
"""
import os
import json

import nff
import torch
from torch.optim import Adam


def get_trainer(args, model, train_loader, val_loader, metrics, loss_fn=None):
    # setup hook and logging
    hooks = [nff.train.MaxEpochHook(args.max_epochs)]

    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    schedule = nff.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    printer = nff.train.PrintingHook(
        os.path.join(args.model_path, 'log'),
        metrics,
        log_memory=(args.device != 'cpu'),
        separator=' | '
    )
    hooks.append(printer)

    if args.logger == 'csv':
        logger = nff.train.CSVHook(
            os.path.join(args.model_path, 'log'),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    elif args.logger == 'tensorboard':
        logger = nff.train.TensorboardHook(
            os.path.join(args.model_path, 'log'),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    if loss_fn is None:
        loss_fn = nff.train.build_mse_loss(json.loads(args.loss_coef))

    trainer = nff.train.Trainer(
        args.model_path,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=1,
        hooks=hooks,
    )
    return trainer

