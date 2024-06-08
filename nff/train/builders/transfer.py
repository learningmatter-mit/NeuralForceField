from nff.train.transfer import ChgnetLayerFreezer, MaceLayerFreezer, PainnDiabatLayerFreezer, PainnLayerFreezer

LAYER_FREEZER_DICT = {
    "CHGNetNFF": ChgnetLayerFreezer,
    "NffScaleMACE": MaceLayerFreezer,
    "PainnDiabat": PainnDiabatLayerFreezer,
    "Painn": PainnLayerFreezer,
}


def get_layer_freezer(model_type: str = "CHGNetNFF"):
    """Get the layer freezer for a given model type.

    Args:
        model_type (str, optional): model type. Defaults to "CHGNetNFF".

    Returns:
        LayerFreezer: layer freezer for the model type

    Raises:
        ValueError: if the model type is not supported
    """
    try:
        model_freezer = LAYER_FREEZER_DICT[model_type]()
    except KeyError as e:
        raise ValueError(
            f"Model type {model_type} not supported. Supported models are {list(LAYER_FREEZER_DICT.keys())}."
        ) from e

    return model_freezer
