"""This module contains functions to predict properties and errors of a model.
These functions specifically are used when doing biased MD simulations with
model uncertainty as the collective variable (CV).

Author: Aik Rui Tan
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch
from ase.atoms import Atoms
from mace import data
from mace.tools import torch_geometric, utils
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import UNDIRECTED, NeuralFF
from nff.utils.cuda import batch_detach, batch_to
from nff.utils.misc import get_atoms

__all__ = [
    "get_atoms",
    "get_errors",
    "get_nff_embedding",
    "get_prediction",
    "get_prediction_and_errors",
]

EMB_KWARGS = {"invariants_only": True, "pooling": "node_mean"}


def get_nff_prediction(
    model, dset: DataLoader | Dataset | List[Atoms], batch_size: int = 10, device: str = "cuda", pool_embedding=False
) -> dict:
    needs_directed = not any(isinstance(model, i) for i in UNDIRECTED)
    print(type(dset))
    if isinstance(dset, (Dataset, DataLoader)):
        if isinstance(dset, Dataset):
            loader = DataLoader(
                dset,
                batch_size=batch_size,
                collate_fn=collate_dicts,
            )
        elif isinstance(dset, DataLoader):
            loader = dset
            predicted, num_atoms = [], []
            for batch in loader:
                batch = batch_to(batch, device=device)
                pred = model(batch)
                batch = batch_detach(batch)
                # pred = batch_detach(pred)
                num_atoms.extend(batch["num_atoms"])

                predicted.append(pred)
    if isinstance(dset, list) and all(isinstance(d, Atoms) for d in dset):
        batches = [AtomsBatch.from_atoms(d, directed=needs_directed) for d in dset]
        calc = NeuralFF(
            model, device=device, properties=["energy", "forces", "embedding"], pool_embedding=pool_embedding
        )
        predicted, num_atoms = [], []
        for abatch in batches:
            abatch.calc = calc
            calc.calculate(abatch)
            pred = calc.results
            pred["xyz"] = torch.Tensor(abatch.get_positions())
            pred["xyz"].requires_grad = True
            num_atoms.append(len(abatch))
            predicted.append(pred)

    predicted = {
        k: torch.concat([torch.Tensor(p[k]) if not isinstance(p[k], torch.Tensor) else p[k] for p in predicted])
        for k in predicted[0]
    }
    predicted["num_atoms"] = num_atoms

    return predicted


def get_mace_prediction(
    model,
    dset: Dataset | list[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
):
    atoms_list = [get_atoms(d) for d in dset] if isinstance(dset, Dataset) else dset

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max)) for config in configs],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    _predicted = []
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=requires_grad)
        pred = {
            "energy": output["energy"],
            "energy_grad": -output["forces"],
            "embedding": output["node_feats"],
            "xyz": output["xyz"],
        }

        if not requires_grad:
            pred = batch_detach(pred)

        _predicted.append(pred)

        batch = batch.cpu()
        output = batch_detach(output)
        pred = batch_detach(pred)

    # Concatenate data
    predicted = {}
    for k in _predicted[0]:
        value = [p[k] for p in _predicted]
        if k not in ["embedding", "node_feats"]:
            value = torch.cat(value, dim=0)
        else:
            # if the model is an ensemble that has multiple networks
            if hasattr(model, "networks"):
                new_value = [[] for _ in range(len(value[0]))]
                for val in value:
                    for i, v in enumerate(val):
                        new_value[i].append(v)
                new_value = [torch.cat(v) for v in new_value]
                value = new_value
                del new_value
            # if the model is a single model (non-ensemble)
            else:
                value = torch.cat(list(value), dim=0)

        predicted[k] = value

    predicted["num_atoms"] = [len(atoms) for atoms in atoms_list]

    return predicted


def get_prediction(
    model: torch.nn.Module | list[torch.nn.Module],
    dset: Dataset | list[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
    get_target: bool = True,
    **kwargs,
) -> tuple[dict, dict]:
    """Get predictions from a model regardless of its architecture. This function
    should work for both NFF model implementations (PaiNN and SchNet) and MACE models.

    Args:
        model (torch.nn.Module or list of Modules): the model or models to evaluate
        dset (Dataset | List[Atoms]): the dataset to evaluate the model on
        batch_size (int, optional): the number of structures in each batch of predictions.
            Defaults to 10.
        device (str, optional): where to run predictions. Defaults to "cuda".
        requires_grad (bool, optional): whether or not the models require a gradient (usually
            True if the model is being trained). Defaults to False.
        kwargs: additional keyword arguments to pass to the prediction function.

    Returns:
        tuple[dict, dict]: target and predicted values for the energy and forces.
    """
    if "Painn" in model.__repr__() or "SchNet" in model.__repr__():
        predicted = get_nff_prediction(model, dset, batch_size=batch_size, device=device, **kwargs)

    elif "MACE" in model.__repr__():
        predicted = get_mace_prediction(
            model, dset, batch_size=batch_size, device=device, requires_grad=requires_grad, **kwargs
        )

    if get_target:
        if isinstance(dset, Dataset):
            target = {
                "energy": dset.props["energy"],
                "energy_grad": dset.props["energy_grad"],
            }
        elif isinstance(dset, DataLoader):
            target = {
                "energy": dset.dataset.props["energy"],
                "energy_grad": dset.dataset.props["energy_grad"],
            }
        else:
            target = {
                "energy": np.array([at.get_potential_energy() for at in dset]),
                "energy_grad": np.concatenate([-at.get_forces(apply_constraint=False) for at in dset]),
            }

        print(target["energy"].shape, target["energy_grad"])
        target["energy"] = torch.tensor(target["energy"]).to(predicted["energy"].device)
        target["energy_grad"] = torch.tensor(target["energy_grad"]).to(predicted["energy_grad"].device)

    else:
        target = None

    return target, predicted


def get_errors(
    predicted: dict, target: dict, mae: bool = True, rmse: bool = True, r2: bool = True, max_error: bool = True
) -> dict:
    """Get errors between predicted and target values.

    Args:
        predicted (dict): the predicted values
        target (dict): the target values
        mae (bool, optional): whether to compute the mean absolute error. Defaults to True.
        rmse (bool, optional): whether to compute the root mean squared error. Defaults to True.
        r2 (bool, optional): whether to compute the R^2 score. Defaults to True.
        max_error (bool, optional): whether to compute the maximum error. Defaults to True.

    Returns:
        dict: dictionary of errors
    """
    pred_energy = predicted["energy"].detach().cpu().numpy()
    targ_energy = target["energy"].detach().cpu().numpy()

    pred_forces = -predicted["energy_grad"].detach().cpu().numpy()
    targ_forces = -target["energy_grad"].detach().cpu().numpy()

    if pred_energy.ndim > 1 and pred_energy.shape != targ_energy.shape:
        pred_energy = pred_energy.mean(-1)
    if pred_forces.ndim > 2 and pred_forces.shape != targ_forces.shape:
        pred_forces = pred_forces.mean(-1)

    errors = {"energy": {}, "forces": {}}
    if mae:
        mae_energy = np.mean(np.abs(pred_energy - targ_energy))
        mae_forces = np.mean(np.abs(pred_forces - targ_forces))
        errors["energy"]["mae"] = mae_energy
        errors["forces"]["mae"] = mae_forces

    if rmse:
        rmse_energy = np.sqrt(np.mean((pred_energy - targ_energy) ** 2))
        rmse_forces = np.sqrt(np.mean((pred_forces - targ_forces) ** 2))
        errors["energy"]["rmse"] = rmse_energy
        errors["forces"]["rmse"] = rmse_forces

    if r2:
        r2_energy = 1 - np.sum((pred_energy - targ_energy) ** 2) / np.sum((targ_energy - np.mean(targ_energy)) ** 2)
        r2_forces = 1 - np.sum((pred_forces - targ_forces) ** 2) / np.sum((targ_forces - np.mean(targ_forces)) ** 2)
        errors["energy"]["r2"] = r2_energy
        errors["forces"]["r2"] = r2_forces

    if max_error:
        max_error_energy = np.max(np.abs(pred_energy - targ_energy))
        max_error_forces = np.max(np.abs(pred_forces - targ_forces))
        errors["energy"]["max_error"] = max_error_energy
        errors["forces"]["max_error"] = max_error_forces

    return errors


def get_nff_embedding(
    model: torch.nn.Module | list[torch.nn.Module], dset: Dataset, batch_size: int, device: str
) -> torch.Tensor:
    """Get the atom embeddings from a model.

    Args:
        model (torch.nn.Module or list of Modules): the model or models to evaluate
        dset (Dataset): the dataset to evaluate the model on
        batch_size (int): the number of structures in each batch of predictions.
        device (str): where to run predictions.

    Returns:
        torch.Tensor: the atom embeddings from the model.
    """
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
        shuffle=False,
    )

    embedding = []
    for batch in loader:
        batch = batch_to(batch, device=device)
        emb = torch.stack([m.atomwise(batch)[0]["embedding"] for m in model])
        emb = emb.detach().cpu()
        batch = batch_detach(batch)

        embedding.append(emb)

    embedding = torch.concat(embedding, dim=1)

    return embedding  # (n_networks, n_atoms, n_atom_basis)


def get_prediction_and_errors(
    model: torch.nn.Module | list[torch.nn.Module], dset: Dataset | list[Atoms], batch_size: int, device: str
) -> tuple[dict, dict, dict]:
    """Get predictions and errors from a model.

    Args:
        model (torch.nn.Module or list of Modules): the model or models to evaluate
        dset (Dataset | list[Atoms]): the dataset to evaluate the model on
        batch_size (int): the number of structures in each batch of predictions.
        device (str): where to run predictions.

    Returns:
        tuple[dict, dict, dict]: target values, predicted values, and errors.
    """
    target, predicted = get_prediction(model, dset, batch_size, device)

    target = batch_detach(target)
    predicted = batch_detach(predicted)

    errors = get_errors(predicted, target, mae=True, rmse=True, r2=True, max_error=True)

    return target, predicted, errors


def get_residual(
    targ: dict,
    pred: dict,
    num_atoms: list[int],
    quantity: str = "energy_grad",
    order: str = "system_mean",
) -> torch.Tensor:
    """Get the residual of the predicted and target quantities

    Args:
        targ (dict): the target quantities
        pred (dict): the predicted quantities
        num_atoms (list[int]): the number of atoms in each molecule
        quantity (str, optional): the quantity to get the residual of. Defaults to "energy_grad".
        order (str, optional): the order of the residual. Defaults to "system_mean".

    Returns:
        torch.Tensor: the residual
    """
    if pred[quantity].shape != targ[quantity].shape:
        pred[quantity] = pred[quantity].mean(-1)

    res = targ[quantity] - pred[quantity]
    res = abs(res)

    if quantity == "energy":
        return res

    # force norm
    res = torch.linalg.norm(res, dim=-1)

    # get residual based on the order
    splits = torch.split(res, num_atoms)
    res = torch.stack(splits, dim=0)

    if "local" in order and "system" not in order:
        device = res.device
        dtype = res.dtype
        size = res.size(0)

        nbr_list = pred["nbr_list"].to(device)

        nbr_count = torch.zeros(size, dtype=dtype, device=device)
        nbr_count.scatter_add_(0, nbr_list[:, 0], torch.ones(nbr_list.size(0), dtype=dtype, device=device))
        nbr_count.scatter_add_(0, nbr_list[:, 1], torch.ones(nbr_list.size(0), dtype=dtype, device=device))

        res_sum = torch.zeros(size, dtype=dtype, device=device)
        res_sum.scatter_add_(0, nbr_list[:, 0], res[nbr_list[:, 1]])
        res_sum.scatter_add_(0, nbr_list[:, 1], res[nbr_list[:, 0]])

        if "local_mean" in order:
            local_res = res_sum / nbr_count
        elif "local_sum" in order:
            local_res = res_sum
        else:
            raise ValueError(f"Invalid order {order}")

        # reshape the res to (num_systems, num_atoms)
        splits = torch.split(local_res, list(num_atoms))
        res = torch.stack(splits, dim=0)

    if "system" in order:
        if "system_mean" in order:
            res = torch.stack([i.mean() for i in res])
        elif "system_sum" in order:
            res = torch.stack([i.sum() for i in res])
        elif "system_max" in order:
            res = torch.stack([i.max() for i in res])
        elif "system_min" in order:
            res = torch.stack([i.min() for i in res])
        elif "system_mean_squared" in order:
            res = torch.stack([(i**2).mean() for i in res])
        elif "system_root_mean_squared" in order:
            res = torch.stack([torch.sqrt((i**2).mean()) for i in res])
        else:
            raise ValueError(f"Invalid order {order}")

    else:
        raise ValueError(f"Invalid order {order}")

    return res


def evaluate_mace(
    model,
    dset,
    batch_size: int = 32,
    output_keys: List[str] = ["energy", "forces", "embeddings"],
    device: str = "cuda",
    embedding_kwargs: Dict[str, Union[bool, int, str]] = EMB_KWARGS,  # noqa
):
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    model.to(device)
    model.eval()
    target_keys = [val for val in output_keys if val != "embeddings" and val != "count"]
    target = {key: [] for key in target_keys}
    output_keys.extend(["count"])
    prediction = {key: [] for key in output_keys}
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=True,
            compute_stress=False,
        )
        if "embeddings" in output_keys:
            descriptors = output["node_feats"]
            processed_embeddings = model.process_embeddings(
                batch_dict,
                descriptors,
                invariants_only=embedding_kwargs["invariants_only"],
                num_layers=embedding_kwargs.get("num_layers", -1),
                pooling=embedding_kwargs["pooling"],
            )
            if embedding_kwargs["pooling"] == "atomic":
                for embeddings_dict in processed_embeddings:
                    prediction["embeddings"].append(embeddings_dict["embeddings"].detach().cpu())
                    prediction["count"].append(embeddings_dict["num_atoms"].detach().cpu())
            elif "node" in embedding_kwargs["pooling"] or embedding_kwargs["pooling"] == "atomic":
                for embeddings_dict in processed_embeddings:
                    for val in embeddings_dict["embeddings"].values():
                        prediction["embeddings"].append(val.detach().cpu().unsqueeze(0))
                    prediction["count"].append(embeddings_dict["num_elements"].detach().cpu().unsqueeze(0))
            else:
                prediction["embeddings"].append(processed_embeddings.detach().cpu())
                count = batch.ptr[1:] - batch.ptr[:-1]  # [n_graphs,]
                prediction["count"].append(count.detach().cpu())
        else:
            count = batch.ptr[1:] - batch.ptr[:-1]  # [n_graphs,]
            prediction["count"].append(count.detach().cpu())

        if "embeddings" in output_keys:
            if "forces" in target_keys:
                pred_forces_processed = model.get_nodewise_pooling(
                    batch_dict, output["forces"], input_key="forces", pooling=embedding_kwargs["pooling"]
                )
                targ_forces_processed = model.get_nodewise_pooling(
                    batch_dict, batch_dict["forces"], input_key="forces", pooling=embedding_kwargs["pooling"]
                )
                if embedding_kwargs["pooling"] == "atomic":
                    for pred_forces_dict in pred_forces_processed:
                        prediction["forces"].append(pred_forces_dict["forces"].detach().cpu())
                    for targ_forces_dict in targ_forces_processed:
                        target["forces"].append(targ_forces_dict["forces"].detach().cpu())
                elif "node" in embedding_kwargs["pooling"]:
                    for pred_forces_dict in pred_forces_processed:
                        for val in pred_forces_dict["forces"].values():
                            prediction["forces"].append(val.detach().cpu().unsqueeze(0))
                    for targ_forces_dict in targ_forces_processed:
                        for val in targ_forces_dict["forces"].values():
                            target["forces"].append(val.detach().cpu().unsqueeze(0))
            for key in target_keys:
                if key != "forces":
                    prediction[key].append(output[key].detach().cpu())
                    target[key].append(batch_dict[key].detach().cpu())
        else:
            for key in target_keys:
                prediction[key].append(output[key].detach().cpu())
                target[key].append(batch_dict[key].detach().cpu())
    del batch
    del batch_dict
    prediction = {k: torch.cat(prediction[k], dim=0) for k in output_keys}
    target = {k: torch.cat(target[k], dim=0) for k in target_keys}
    return target, prediction
