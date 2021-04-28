"""
The transfer learning module provides functions to fine tune
    a pretrained model with a new given dataloader. It relies
    on pretrained models, which can be loaded from checkpoints
    or best models using the Trainer class.
"""


def freeze_parameters(model):
    """
    Freezes all parameters from a given model.

    Args:
        model (any of nff.nn.models)
    """
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = True


def unfreeze_readout(model):
    """
    Unfreezes the parameters from the readout layers.

    Args:
        model (any of nff.nn.models)
    """

    unfreeze_parameters(model.atomwisereadout)


def unfreeze_painn_readout(model,
                           freeze_skip):

    num_readouts = len(model.readout_blocks)
    unfreeze_skip = not freeze_skip

    for i, block in enumerate(model.readout_blocks):
        if unfreeze_skip:
            unfreeze_parameters(block)
        elif (i == num_readouts - 1):
            unfreeze_parameters(block)


def unfreeze_diabat_readout(model,
                            freeze_gap_embedding):
    cross_talk = model.diabatic_readout.cross_talk
    unfreeze_gap = not freeze_gap_embedding
    if not cross_talk:
        return
    for module in cross_talk.coupling_modules:
        if hasattr(module, "readout"):
            unfreeze_parameters(module.readout)
        if (hasattr(module, "featurizer")
                and unfreeze_gap):
            unfreeze_parameters(module.featurizer)


def unfreeze_painn_pooling(model):
    for module in model.pool_dic.values():
        unfreeze_parameters(module)


def painn_diabat_tl(model,
                    freeze_gap_embedding,
                    freeze_pooling,
                    freeze_skip,
                    **kwargs):

    freeze_parameters(model)
    unfreeze_painn_readout(model=model,
                           freeze_skip=freeze_skip)
    unfreeze_diabat_readout(model=model,
                            freeze_gap_embedding=freeze_gap_embedding)

    unfreeze_pool = not freeze_pooling
    if unfreeze_pool:
        unfreeze_painn_pooling(model)


def painn_tl(model,
             freeze_pooling,
             freeze_skip,
             **kwargs):

    freeze_parameters(model)
    unfreeze_painn_readout(model=model,
                           freeze_skip=freeze_skip)
    unfreeze_pool = not freeze_pooling
    if unfreeze_pool:
        unfreeze_painn_pooling(model)
