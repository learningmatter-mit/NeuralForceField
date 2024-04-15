import os
import urllib
from typing import Iterable, Union, List

import torch
from mace.calculators.mace import get_model_dtype
from mace.data.atomic_data import AtomicNumberTable
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.radial import BesselBasis, GaussianBasis


# get the path to NFF models dir, which is the parent directory of this file
module_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "models")
)
print(module_dir)
LOCAL_MODEL_PATH = os.path.join(
    module_dir, "foundation_models/mace/2023-12-03-mace-mp.model"
)

MACE_URLS = dict(
    small="http://tinyurl.com/46jrkm3v",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
    medium="http://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
    large="http://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
)


def get_mace_mp_model_path(model: str = None) -> str:
    """Get the default MACE MP model. Replicated from the MACE codebase,
    Copyright (c) 2022 ACEsuit/mace and licensed under the MIT license.

    Args:
        model (str, optional): MACE_MP model that you want to get.
            Defaults to None. Can be "small", "medium", "large", or a URL.

    Raises:
        RuntimeError: raised if the model download fails and no local model is found

    Returns:
        str: path to the model
    """
    if model in (None, "medium") and os.path.isfile(LOCAL_MODEL_PATH):
        model_path = LOCAL_MODEL_PATH
        print(
            f"Using local medium Materials Project MACE model for MACECalculator {model}"
        )
    elif model in (None, "small", "medium", "large") or str(model).startswith("https:"):
        try:
            checkpoint_url = (
                MACE_URLS.get(model, MACE_URLS["medium"])
                if model in (None, "small", "medium", "large")
                else model
            )
            cache_dir = os.path.expanduser("~/.cache/mace")
            checkpoint_url_name = "".join(
                c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
            )
            model_path = f"{cache_dir}/{checkpoint_url_name}"
            if not os.path.isfile(model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {checkpoint_url!r}")
                urllib.request.urlretrieve(checkpoint_url, model_path)
                print(f"Cached MACE model to {model_path}")
            msg = f"Loading Materials Project MACE with {model_path}"
            print(msg)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc
    else:
        raise RuntimeError(
            "Model download failed and no local model found"
        )

    return model_path


def get_init_kwargs_from_model(model: Union[ScaleShiftMACE, MACE]) -> dict:
    """Get the initialization arguments from the model"""
    if isinstance(model.radial_embedding.bessel_fn, BesselBasis):
        radial_type = "bessel"
    elif isinstance(model.radial_embedding.bessel_fn, GaussianBasis):
        radial_type = "gaussian"
    
    init_kwargs = {
        "r_max": model.r_max.item(),
        "num_bessel": model.radial_embedding.out_dim,
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics.irreps_out.lmax,
        "interaction_cls": type(model.interactions[1]),
        "interaction_cls_first": type(model.interactions[0]),
        "num_interactions": model.num_interactions.item(),
        "num_elements": model.node_embedding.linear.irreps_in.dim,
        "hidden_irreps": model.interactions[0].hidden_irreps,
        "MLP_irreps": model.readouts[-1].hidden_irreps,
        "atomic_energies": model.atomic_energies_fn.atomic_energies,
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers.tolist(),
        "correlation": model.products[0]
        .symmetric_contractions.contractions[0]
        .correlation,
        "gate": model.readouts[-1].non_linearity.acts[0].f,
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "radial_type": radial_type
    }
    if isinstance(model, ScaleShiftMACE):
        init_kwargs.update({"atomic_inter_scale":model.scale_shift.scale, "atomic_inter_shift": model.scale_shift.shift})

    return init_kwargs


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))

