import logging
import os
import urllib
from pathlib import Path
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mace.data.atomic_data import AtomicNumberTable
from mace.modules.blocks import AtomicEnergiesBlock
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.radial import BesselBasis, GaussianBasis, AgnesiTransform, SoftTransform
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.data import Data
from torch import Tensor
from e3nn import o3
from nff.data import Dataset
from nff.utils.cuda import detach
from mace.calculators.foundations_models import download_mace_mp_checkpoint, mace_mp_names

# get the path to NFF models dir, which is the parent directory of this file
module_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "models"))
print(module_dir)
LOCAL_MODEL_PATH = os.path.join(module_dir, "foundation_models/mace/2023-12-03-mace-mp.model")

MACE_URLS = dict(
    small="http://tinyurl.com/46jrkm3v",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
    medium="http://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
    large="http://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
)


def _check_non_zero(std):
    if std == 0.0:
        logging.warning("Standard deviation of the scaling is zero, Changing to no scaling")
        std = 1.0
    return std


def get_mace_foundtion_model_path(model: Optional[str] = None, supress_print=True) -> str:
    """Get the default MACE MP model. Replicated from the MACE codebase,
    Copyright (c) 2022 ACEsuit/mace and licensed under the MIT license.

    Args:
        model (str, optional): MACE_MP model that you want to get.
            Defaults to None. Can be "small", "medium", "large", or a URL.
        supress_print (bool, optional): Whether to suppress print statements. Defaults to True.

    Raises:
        RuntimeError: raised if the model download fails and no local model is found

    Returns:
        str: path to the model
    """
    try:
        if model in mace_mp_names or str(model).startswith("https:"):
            model_path = download_mace_mp_checkpoint(model)
        else:
            if not Path(model).exists():
                raise FileNotFoundError(f"{model} not found locally")
            model_path = model
        if not supress_print:
            print(f"Using MACE mdoel with {model_path}")
    except Exception as exc:
        raise RuntimeError("Model download failed or no local model found") from exc

    return model_path


def get_init_kwargs_from_model(model: Union[ScaleShiftMACE, MACE]) -> dict:
    """Get the initialization arguments from the model"""
    if isinstance(model.radial_embedding.bessel_fn, BesselBasis):
        radial_type = "bessel"
    elif isinstance(model.radial_embedding.bessel_fn, GaussianBasis):
        radial_type = "gaussian"
    try:
        if isinstance(model.radial_embedding.distance_transform, AgnesiTransform):
            distance_transform = "Agnesi"
        elif isinstance(model.radial_embedding.distance_transform, SoftTransform):
            distance_transform = "Soft"
    except:
        distance_transform = None
    try:
        heads = model.heads
    except:
        heads = ["default"]
    try:
        pair_repulsion = model.pair_repulsion
    except:
        pair_repulsion = False
    num_interactions = model.num_interactions.item()
    MLP_irreps = []

    # Iterate over the irreducible representations in the model
    for mul, ir in model.readouts[num_interactions-1].hidden_irreps:
        # Adjust the multiplicity by dividing by the number of heads
        new_mul = mul // len(heads)  # Use integer division to avoid float results
        new_ir = ir  # No need to change `ir`, just assign it for clarity

        # Append the new multiplicity and irreducible representation to the list
        MLP_irreps.append((new_mul, new_ir))

    # Convert the list into an `o3.Irreps` object
    MLP_irreps = o3.Irreps(MLP_irreps)
    init_kwargs = {
        "r_max": model.r_max.item(),
        "num_bessel": model.radial_embedding.out_dim,
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics.irreps_out.lmax,
        "interaction_cls": type(model.interactions[1]),
        "interaction_cls_first": type(model.interactions[0]),
        "num_interactions": num_interactions,
        "num_elements": model.node_embedding.linear.irreps_in.dim,
        "hidden_irreps": model.interactions[0].hidden_irreps,
        "MLP_irreps": MLP_irreps,
        "atomic_energies": model.atomic_energies_fn.atomic_energies,
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers.tolist(),
        "correlation": model.products[0].symmetric_contractions.contractions[0].correlation,
        "gate": model.readouts[-1].non_linearity.acts[0].f,
        "pair_repulsion": pair_repulsion,
        "distance_transform": distance_transform,
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "radial_type": radial_type,
        "heads": heads
    }
    if type(model).__name__ == "ScaleShiftMACE":
        init_kwargs.update({"atomic_inter_scale":model.scale_shift.scale, "atomic_inter_shift": model.scale_shift.shift})
    # if isinstance(model, ScaleShiftMACE):

    return init_kwargs


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    """Get the atomic number table from a list of atomic numbers.

    Args:
    zs (Iterable[int]): list of atomic numbers

    Returns:
    AtomicNumberTable: atomic number table
    """
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(z_set))


def compute_average_E0s(train_dset: Dataset, z_table: AtomicNumberTable, desired_units: str = "eV") -> Dict[int, float]:
    """Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s

    Args:
    train_dset (Dataset): dataset of training data
    z_table (AtomicNumberTable): table of atomic numbers
    desired_units (str, optional): units for atomic energies. Defaults to "eV".

    Returns:
    Dict[int, float]: dictionary of atomic energies
    """
    original_units = train_dset.units
    train_dset.to_units(desired_units)

    len_train = len(train_dset)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = train_dset[i]["energy"]
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(detach(train_dset[i]["nxyz"][:, 0], to_numpy=True).astype(int) == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning("Failed to compute E0s using least squares regression, using the same for all atoms")
        atomic_energies_dict = {}
        for z in z_table.zs:
            atomic_energies_dict[z] = 0.0

    train_dset.to_units(original_units)
    return atomic_energies_dict


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
    z_table: AtomicNumberTable,
) -> Tuple[float, float]:
    """Compute the mean of atomic energies and RMS of forces for a dataset.

    Args:
    data_loader (torch.utils.data.DataLoader): data loader
    atomic_energies (np.ndarray): atomic energies
    z_table (AtomicNumberTable): table of atomic numbers

    Returns:
    Tuple[float, float]: mean and RMS of forces
    """

    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)
    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        # obtain one hot encoded atoms based on z_table atom indices
        zs = detach(batch["nxyz"][:, 0], to_numpy=True).astype(int)
        one_hot_zs = torch.zeros((len(zs), len(z_table)), dtype=torch.get_default_dtype())
        for i, z in enumerate(zs):
            one_hot_zs[i, z_table.z_to_index(z)] = 1
        # compute atomic energies
        node_e0 = atomic_energies_fn(one_hot_zs)
        graph_sizes = batch["num_atoms"]  # list of num atoms

        # given graph_sizes, transform to list of indices
        # index starts from 0, denoting the first graph
        # each index is repeated by the number of atoms in the graph
        counter = 0
        batch_indices = torch.zeros(sum(graph_sizes), dtype=torch.long)
        for i, size in enumerate(graph_sizes):
            batch_indices[counter : counter + size] = i
            counter += size

        # get the graph energy
        graph_e0s = scatter_sum(src=node_e0, index=batch_indices, dim=-1, dim_size=len(graph_sizes))
        atom_energy_list.append((batch["energy"] - graph_e0s) / graph_sizes)  # {[n_graphs], }
        forces_list.append(-batch["energy_grad"])  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = detach(torch.mean(atom_energies), to_numpy=True).item()
    rms = detach(torch.sqrt(torch.mean(torch.square(forces))), to_numpy=True).item()
    rms = _check_non_zero(rms)
    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    """Compute the average number of neighbors in a dataset.

    Args:
    data_loader (torch.utils.data.DataLoader): data loader

    Returns:
    float: average number of neighbors
    """
    num_neighbors = []

    for batch in data_loader:
        unique_neighbors_list = torch.unique(batch["nbr_list"], dim=0)  # remove repeated neighbors
        receivers = unique_neighbors_list[:, 1]

        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype()))
    return detach(avg_num_neighbors, to_numpy=True).item()


def update_mace_init_params(
    train: Dataset,
    val: Dataset,
    train_loader: torch.utils.data.DataLoader,
    model_params: Dict,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[int, float, np.ndarray, List[int]]]:
    """Update the MACE model initialization parameters based values obtained from training and validation datasets.

    Args:
    train (Dataset): training dataset
    val (Dataset): validation dataset
    train_loader (torch.utils.data.DataLoader): training data loader
    model_params (Dict[str, Union[int, float, np.ndarray, List[int]]): model parameters
    logger (Optional[logging.Logger], optional): logger. Defaults to None.

    Returns:
    Tuple: updated model parameters
    """
    if not logger:
        logger = logging.getLogger(__name__)

    # z_table
    z_table = get_atomic_number_table_from_zs(
        [
            int(z)
            for data_split in (train, val)
            for data in data_split
            for z in detach(data["nxyz"][:, 0], to_numpy=True)
        ]
    )
    logger.info("Z Table %s", z_table.zs)

    # avg_num_neighbors
    # Average number of neighbors: 41.22802734375
    # BUG: doesn't really match but might not matter!
    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    logger.info("Average number of neighbors: %s", avg_num_neighbors)

    # atomic_energies
    # {8: -4.930998234144857, 38: -5.8572783662579795, 77: -8.316066722236071}
    atomic_energies_dict = compute_average_E0s(train, z_table)
    atomic_energies: np.ndarray = np.array([atomic_energies_dict[z] for z in z_table.zs])
    logger.info("Atomic energies: %s", atomic_energies.tolist())

    # mean & std
    # Mean and std of atomic energies: -0.0014447236899286509, 7.5926432609558105
    atomic_inter_shift, atomic_inter_scale = compute_mean_rms_energy_forces(train_loader, atomic_energies, z_table)
    logger.info("Mean and std of atomic energies: %s, %s", atomic_inter_shift, atomic_inter_scale)

    model_params["atomic_inter_scale"] = atomic_inter_scale
    model_params["atomic_inter_shift"] = atomic_inter_shift
    model_params["num_elements"] = len(z_table)
    model_params["atomic_energies"] = atomic_energies
    model_params["avg_num_neighbors"] = avg_num_neighbors
    model_params["atomic_numbers"] = z_table.zs

    return model_params


class NffBatch(Batch):
    def __init__(self, batch=None, ptr=None, **kwargs):
        super().__init__(batch=batch, ptr=ptr, **kwargs)

    def get_example(self, idx: int) -> Data:
        r"""Reconstructs the :class:`torch_geometric.data.Data` object at index
        :obj:`idx` from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the batch "
                "object was not created using `Batch.from_data_list()`."
            )

        data = {}
        idx = self.num_graphs + idx if idx < 0 else idx

        for key in self.__slices__:
            item = self[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item[start:end]
                    item = item[0] if len(item) == 1 else item

            # Decrease its value by `cumsum` value:
            cum = self.__cumsum__[key][idx]
            if isinstance(item, Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum

            data[key] = item
        elems = data.pop("elems")
        data = Data(**data)
        data.elems = elems
        if self.__num_nodes_list__[idx] is not None:
            data.num_nodes = self.__num_nodes_list__[idx]

        return data
