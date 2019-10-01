"""
The transfer learning module provides functions to fine tune
    a pretrained model with a new given dataloader. It relies
    on pretrained models, which can be loaded from checkpoints
    or best models using the Trainer class.
"""

import torch


def freeze_parameters(model):
    """
    Freezes all parameters from a given model.

    Args:
        model (any of nff.nn.models)
    """
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_readout(model):
    """
    Unfreezes the parameters from the readout layers.

    Args:
        model (any of nff.nn.models)
    """
    for param in model.atomwisereadout.parameters():
        param.requires_grad = True

