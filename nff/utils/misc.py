"""Miscellaneous utility functions for NFF"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.build.rotate import rotation_matrix_from_points
from sklearn.metrics import (
    accuracy_score,
    auc,
    log_loss,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from tqdm import tqdm

# optimization goal for various metrics

METRIC_DIC = {
    "pr_auc": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "class_loss": "minimize",
    "regress_loss": "minimize",
    "mae": "minimize",
    "mse": "minimize",
}

METRICS = list(METRIC_DIC.keys())

# transform from chemprop syntax to our syntax for the metrics

CHEMPROP_TRANSFORM = {
    "auc": "roc_auc",
    "prc-auc": "pr_auc",
    "binary_cross_entropy": "class_loss",
    "mse": "regress_loss",
}

# metrics available in chemprop

CHEMPROP_METRICS = [
    "auc",
    "prc-auc",
    "rmse",
    "mae",
    "mse",
    "r2",
    "accuracy",
    "cross_entropy",
    "binary_cross_entropy",
]


def tqdm_enum(iterable: Iterable) -> tuple[int, any]:  # type: ignore
    """Wrap tqdm around `enumerate`.

    Args:
        iterable (Iterable): an iterable (e.g. list)

    Returns:
        i (int): current index
        y: current value
    """

    i = 0
    for y in tqdm(iterable):
        yield i, y
        i += 1


def log(prefix: str, msg: str) -> None:
    """Logs on the screen a message in the format of 'PREFIX:  msg'

    Args:
        prefix (str): prefix of the message
        msg (str): message to log
    """
    print(f"{prefix.upper():>12}:  {msg}")


def add_json_args(args: argparse.ArgumentParser, config_flag: str = "config_file") -> argparse.Namespace:
    """Add arguments from a JSON file to the arguments.

    Args:
        args (argparse.ArgumentParser): arguments
        config_flag (str): name of the arg key
            that gives the name of the config file.

    Returns:
        args (argparse.Namespace): arguments
    """
    config_path = getattr(args, config_flag, None)
    if config_path is not None:
        with open(config_path, "r") as f:
            config_args = json.load(f)
        if "details" in config_args:
            config_args.update(config_args["details"])
            config_args.pop("details")
        for key, val in config_args.items():
            if hasattr(args, key):
                setattr(args, key, val)
    return args


def parse_args(parser: argparse.ArgumentParser, config_flag: str = "config_file") -> argparse.Namespace:
    """Parse arguments.

    Args:
        parser (argparse.ArgumentParser): argument parser
        config_flag (str): name of the arg key
            that gives the name of the config file.

    Returns:
        args (argparse.Namespace): arguments
    """
    # parse the arguments
    args = parser.parse_args()

    # if the config path is specified, then load
    # arguments from that file and apply the results
    # to `args`

    args = add_json_args(args, config_flag=config_flag)
    return args


def fprint(msg: str) -> None:
    """Print a string immediately.

    Args:
        msg (str): string to print

    Returns:
        None
    """
    print(msg)
    sys.stdout.flush()


def bash_command(cmd: str) -> subprocess.Popen:
    """Run a command from the command line using subprocess.

    Args:
        cmd (str): command

    Returns:
        None
    """
    return subprocess.Popen(cmd, shell=True, executable="/bin/bash")


def convert_metric(metric: str) -> str:
    """Convert a metric name to a fixed name that can be used in
    various scripts.

    Args:
        metric (str): input metric

    Returns:
        metric (str): output metric
    """
    if metric in ["prc_auc", "prc-auc"]:
        metric = "pr_auc"
    elif metric in ["auc", "roc-auc"]:
        metric = "roc_auc"
    return metric


def prepare_metric(lines: list[str], metric: str) -> tuple[int, float, int, str]:
    """Get various metric quantities before parsing a log fine.

    Args:
        lines (list[str]): lines in the log file
        metric (str): name of metric

    Returns:
        idx (int): index at which the metric score occurs
            when the given line has been split by `|`
        best_score (float): initial best score
        best_epoch (int): initial best_epoch
        optim (str): goal of the metric optimization (i.e.
            minimize or maximize.)
    """
    header_items = [i.strip() for i in lines[0].split("|")]
    metric = convert_metric(metric)
    if "loss" in metric:
        idx = header_items.index("Validation loss")
    else:
        for i, item in enumerate(header_items):
            sub_keys = metric.split("_")
            if all(key.lower() in item.lower() for key in sub_keys):
                idx = i

    optim = METRIC_DIC[metric]

    best_score = float("inf") if optim == "minimize" else -float("inf")

    best_epoch = -1

    return idx, best_score, best_epoch, optim


def parse_score(model_path: str, metric: str) -> tuple[float, int]:
    """Find the best score and best epoch according to a given metric.

    Args:
        model_path (str): path to the training folder
        metric (str): name of metric

    Returns:
        best_score (float): best validation score
        best_epoch (int): epoch with the best validation score
    """
    log_path = os.path.join(model_path, "log_human_read.csv")
    with open(log_path, "r") as f:
        lines = f.readlines()

    idx, best_score, best_epoch, optim = prepare_metric(lines=lines, metric=metric)

    for line in lines:
        splits = [i.strip() for i in line.split("|")]
        try:
            score = float(splits[idx])
        except (ValueError, IndexError):
            continue

        if any(
            [
                (optim == "minimize" and score < best_score),
                (optim == "maximize" and score > best_score),
            ]
        ):
            best_score = score
            best_epoch = splits[1]

    return best_score, best_epoch


def read_csv(path: str):
    """Read a csv into a dictionary.

    Args:
        path (str): path to the csv file

    Returns:
        dic (dict): dictionary version of the file
    """
    with open(path, "r") as f:
        lines = f.readlines()

    keys = lines[0].strip().split(",")
    dic = {key: [] for key in keys}
    for line in lines[1:]:
        vals = line.strip().split(",")
        for key, val in zip(keys, vals):
            if val.isdigit():
                dic[key].append(int(val))
            else:
                try:
                    dic[key].append(float(val))
                except ValueError:
                    dic[key].append(val)
    return dic


def write_csv(path: str, dic: dict) -> None:
    """Write a dictionary to a csv.

    Args:
        path (str): path to the csv file
        dic (dict): dictionary

    Returns:
        None
    """
    keys = sorted(dic.keys())
    if "smiles" in keys:
        keys.remove("smiles")
        keys.insert(0, "smiles")

    lines = [",".join(keys)]
    for idx in range(len(dic[keys[0]])):
        vals = [dic[key][idx] for key in keys]
        line = ",".join(str(val) for val in vals)
        lines.append(line)
    text = "\n".join(lines)

    with open(path, "w") as f:
        f.write(text)


def prop_split(max_specs: int, dataset_type: str, props: list[str], sample_dic: dict, seed: int) -> list[str]:
    """Sample a set of smiles strings by up to a maximum number. If the
    property of interest is a binary value, try to get as many of the
    underrepresented class as possible.

    Args:
        max_specs (int): maximum number of species
        dataset_type (str): type of problem (classification or regression)
        props (list[str]): names of properties you'll be fitting
        sample_dic (dict): dictionary of the form {smiles: sub_dic} for the
            set of smiles strings, where sub_dic contains other information,
            e.g. about `props`.
        seed (int): random seed for sampling

    Returns:
        keep_smiles (list[str]): sampled smiles strings.
    """
    random.seed(seed)

    if max_specs is not None and dataset_type == "classification":
        msg = "Not implemented for multiclass"
        assert len(props) == 1, msg

        prop = props[0]
        pos_smiles = [key for key, sub_dic in sample_dic.items() if sub_dic.get(prop) == 1]
        neg_smiles = [key for key, sub_dic in sample_dic.items() if sub_dic.get(prop) == 0]

        # find the underrepresnted and overrepresented class
        if len(pos_smiles) < len(neg_smiles):
            underrep = pos_smiles
            overrep = neg_smiles
        else:
            underrep = neg_smiles
            overrep = pos_smiles

        # if possible, keep all of the underrepresented class
        if max_specs >= 2 * len(underrep):
            random.shuffle(overrep)
            num_left = max_specs - len(underrep)
            keep_smiles = underrep + overrep[:num_left]

        # otherwise create a dataset with half of each
        else:
            random.shuffle(underrep)
            random.shuffle(overrep)
            keep_smiles = underrep[: max_specs // 2] + overrep[max_specs // 2 :]
    else:
        keep_smiles = list(sample_dic.keys())

        # if setting a maximum, need to shuffle in order
        # to take random smiles

        if max_specs is not None:
            random.shuffle(keep_smiles)

    if max_specs is not None:
        keep_smiles = keep_smiles[:max_specs]

    return keep_smiles


def get_split_names(train_only: bool, val_only: bool, test_only: bool) -> list[str]:
    """Get names of dataset splits.

    Args:
      train_only (bool): only load the training set
      val_only (bool): only load the validation set
      test_only (bool): only load the test set

    Returns:
        names (list[str]): names of splits
            (train, val, and/or test) that we're
            monitoring.
    """
    only_dic = {"train": train_only, "val": val_only, "test": test_only}

    requested = [name for name, only in only_dic.items() if only]
    if len(requested) > 1:
        string = ", ".join(requested)
        msg = f"Requested {string}, which are mutually exclusive"
        raise Exception(msg)

    names = requested if len(requested) != 0 else ["train", "val", "test"]

    return names


def preprocess_class(pred):
    """Preprocess classifier predictions. This applies,
    for example, if you train an sklearn regressor
    rather than classifier, which doesn't necessarily
    predict a value between 0 and 1.

    Args:
        pred (np.array or torch.Tensor or list): predictions

    Returns:
        pred (np.array or torch.Tensor or list): predictions
            with max 1 and min 0.
    """
    to_list = False
    if isinstance(pred, list):
        pred = np.array(pred)
        to_list = True

    # make sure the min and max are 0 and 1
    pred[pred < 0] = 0
    pred[pred > 1] = 1

    if to_list:
        pred = pred.tolist()

    return pred


def apply_metric(metric: str, pred: Iterable, actual: Iterable) -> float:
    """Apply a metric to a set of predictions.

    Args:
      metric (str): name of metric
      pred (iterable): predicted values
      actual (iterable): actual values

    Returns:
      score (float): metric score
    """
    if metric == "auc":
        pred = preprocess_class(pred)
        score = 0 if max(pred) == 0 else roc_auc_score(y_true=actual, y_score=pred)
    elif metric == "prc-auc":
        pred = preprocess_class(pred)
        if max(pred) == 0:
            score = 0
        else:
            precision, recall, _ = precision_recall_curve(y_true=actual, probas_pred=pred)
            score = auc(recall, precision)
    elif metric == "mse":
        score = ((np.array(pred) - np.array(actual)) ** 2).mean()
    elif metric == "rmse":
        score = ((np.array(pred) - np.array(actual)) ** 2).mean() ** 0.5
    elif metric == "mae":
        score = (abs(np.array(pred) - np.array(actual))).mean()
    elif metric == "r2":
        score = r2_score(y_true=actual, y_pred=pred)
    elif metric == "accuracy":
        np_pred = np.array(pred)
        mask = np_pred >= 0.5
        np_pred[mask] = 1
        np_pred[np.bitwise_not(mask)] = 0
        score = accuracy_score(y_true=actual, y_pred=np_pred)
    elif metric in ["cross_entropy", "binary_cross_entropy"]:
        score = log_loss(y_true=actual, y_pred=np_pred)

    return score


def avg_distances(dset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the average distance between all pairs of atoms in a dataset.

    Args:
        dset (nff.nn.data.Dataset): NFF dataset where all the geometries are
            different conformers for one species.

    Returns:
        all_nbrs (torch.Tensor): all pairs of atoms
        avg_d (torch.Tensor): average distance between all pairs of atoms
    """
    # Get the neighbor list that includes the neighbor list of each conformer

    all_nbrs = []
    for nbrs in dset.props["nbr_list"]:
        for pair in nbrs:
            all_nbrs.append(tuple(pair.tolist()))  # noqa
    all_nbrs_tuple = list(set(all_nbrs))

    all_nbrs = torch.LongTensor([list(i) for i in all_nbrs_tuple])

    num_confs = len(dset)
    all_distances = torch.zeros(num_confs, all_nbrs.shape[0])

    for i, batch in enumerate(dset):
        xyz = batch["nxyz"][:, 1:]
        all_distances[i] = (xyz[all_nbrs[:, 0]] - xyz[all_nbrs[:, 1]]).pow(2).sum(1).sqrt()

    weights = dset.props["weights"].reshape(-1, 1)
    avg_d = (all_distances * weights).sum(0)

    return all_nbrs, avg_d


def cat_props(props: dict) -> dict:
    """Concatenate properties in a dictionary.

    Args:
        props (dict): dictionary of properties

    Returns:
        new_props (dict): concatenated properties
    """
    new_props = {}
    for key, val in props.items():
        if isinstance(val, list):
            if isinstance(val[0], torch.Tensor):
                if len(val[0].shape) == 0:
                    new_props[key] = torch.stack(val)
                else:
                    new_props[key] = torch.cat(val)
            else:
                new_props[key] = val
        elif isinstance(val, torch.Tensor):
            new_props[key] = val

    return new_props


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kronecker product of two tensors.

    Args:
        a (torch.Tensor): tensor 1
        b (torch.Tensor): tensor 2

    Returns:
        out (torch.Tensor): kronecker product of a and b
    """
    ein = torch.einsum("ab,cd-> acbd", a, b)
    out = ein.view(a.size(0) * b.size(0), a.size(1) * b.size(1))

    return out


def load_defaults(direc: str, arg_path: str) -> dict:
    """Load default arguments from a JSON file.

    Args:
        direc (str): directory where the JSON file is located
        arg_path (str): path to the JSON file

    Returns:
        default_args (dict): default arguments
    """
    args_path = os.path.join(direc, arg_path)
    with open(args_path, "r") as f:
        default_args = json.load(f)

    return default_args


def parse_args_from_json(arg_path: str, direc: str) -> argparse.Namespace:
    """Parse arguments from a JSON file.

    Args:
        arg_path (str): path to the JSON file
        direc (str): directory where the JSON file is located

    Returns:
        args (argparse.Namespace): arguments
    """
    default_args = load_defaults(arg_path=arg_path, direc=direc)
    description = default_args["description"]

    parser = argparse.ArgumentParser(description=description)
    default_args.pop("description")

    required = parser.add_argument_group(("required arguments (either in the command line or the config file)"))
    optional = parser.add_argument_group("optional arguments")

    for name, info in default_args.items():
        keys = ["default", "choices", "nargs", "action"]
        kwargs = {key: info[key] for key in keys if key in info}
        if "type" in info:
            kwargs.update({"type": eval(info["type"])})

        # Required arguments get put in one group and optional ones in another
        # so that they're separated in `--help` . We don't actually set
        # required=True for required ones, though, because they can be given in
        # the config file instead of the command line

        group = required if info.get("required", False) else optional
        group.add_argument(f"--{name}", help=info["help"], **kwargs)

    args = parser.parse_args()

    return args


def align_and_return(target: Atoms, atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taken from `ase.build.rotate.minimize_rotation_and_translation`.
    Same as their code, but returns the displacement and the rotation
    matrix.

    Align two structures by minimizing the RMSD.

    Args:
        target (ase.Atoms): target structure
        atoms (ase.Atoms): structure to align

    Returns:
        R (np.ndarray): rotation matrix
        c0 (np.ndarray): center of target structure
        c (np.ndarray): center of structure to align
    """
    p = atoms.get_positions()
    p0 = target.get_positions()

    # centeroids to origin
    c = np.mean(p, axis=0)
    p -= c
    c0 = np.mean(p0, axis=0)
    p0 -= c0

    # Compute rotation matrix
    R = rotation_matrix_from_points(p.T, p0.T)

    atoms.set_positions(np.dot(p, R.T) + c0)

    return R, c0, c


def get_logger(prefix: str) -> callable:  # noqa: D103
    logger = lambda msg: print(f"{time.asctime()}|{prefix}|{msg}")
    return logger


def load_config(path: str) -> dict:
    """Load config file

    Args:
        path (str): path to config file

    Returns:
        dict: config
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        f.close()
    return config


def save_config(config: dict, path: str, **kwargs) -> None:
    """Save config file
    Args:
        config (dict): config
        path (str): path to config file
    """
    with open(path, "w") as f:
        yaml.dump(config, f, **kwargs)
        f.close()


def make_dir(path: str) -> str:
    """Make a directory

    Args:
        path (str): path to directory

    Returns:
        str: path to directory
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def make_and_backup_dir(directory: str) -> None:
    """Make a directory and backup if it already exists

    Args:
        directory (str): directory to make
    """
    if os.path.exists(directory):
        if os.path.exists(f"{directory}_backup"):
            shutil.rmtree(f"{directory}_backup")
        os.rename(directory, f"{directory}_backup")
    os.makedirs(directory)


def get_least_used_device() -> str:
    """Get least used cuda device

    Returns:
        str: cuda device
    """
    import torch

    torch.cuda.init()
    free_memory = [torch.cuda.mem_get_info(device_id) for device_id in range(torch.cuda.device_count())]
    free_memory = [free / total for free, total in free_memory]
    free_memory = [0.0 if i > 1.0 else i for i in free_memory]
    least_used_idx = np.argmax(free_memory).item()
    return f"cuda:{least_used_idx}"


def get_main_model_outdir(config: dict) -> str:
    """Get the main model outdir since now the model outdir can be a list

    Args:
        config (dict): config

    Returns:
        str: outdir
    """
    if isinstance(config["model"]["outdir"], str):
        return config["model"]["outdir"]
    if isinstance(config["model"]["outdir"], list):
        if len(config["model"]["outdir"]) == 1:
            return config["model"]["outdir"][0]

        common_prefix = os.path.commonprefix(config["model"]["outdir"])
        if not common_prefix.endswith("/model/"):
            common_prefix = common_prefix[: common_prefix.rfind("/model/") + 6]
        return common_prefix

    raise ValueError("config['model']['outdir'] is not str or list")


def is_energy_valid(
    energy: float,
    min_energy: float,
    max_energy: float,
) -> bool:
    """Check if the energy is valid

    Args:
        energy: energy
        max_energy: maximum energy (kcal/mol)
        min_energy: minimum energy (kcal/mol)

    Returns:
        True if valid, False otherwise
    """
    return (energy >= min_energy) and (energy < max_energy)


def is_geom_valid(
    atoms: Atoms,
    nbr_list: list | None,
    rmin: float,
    rmax: float | None,
    dist_mat: np.ndarray | None = None,
) -> bool:
    """Check if the atoms object has valid interatomic distances

    Args:
        atoms: ase.Atoms object
        nbr_list: neighbor list
        rmin: minimum distance between atoms (angstrom)
        rmax: maximum distance between atoms (angstrom)
        dist_mat: distance matrix

    Returns:
        True if valid, False otherwise
    """
    # get the distance matrix
    if dist_mat is None:
        dist_mat = atoms.get_all_distances(mic=True)

    # if any two pairs of atoms are too close, return False
    triu_inds = np.triu_indices(dist_mat.shape[0], k=1)
    triu_dist = dist_mat[triu_inds]
    if np.any(triu_dist < rmin):
        return False

    # get distances only for nbr_list (connected bonds)
    # both nbr_list and rmax must be provided since this is a bond check for
    # molecules, does not make sense to check for all atoms
    if nbr_list is not None and rmax is not None:
        distances = dist_mat[nbr_list[:, 0], nbr_list[:, 1]]
        # return False if there are any bond distance greater than rmax
        return not np.any(distances >= rmax)

    return True


def convert_0_2pi_to_negpi_pi(x: float) -> float:
    """Convert angle from 0 to 2pi to -pi to pi for an input in degrees

    Args:
        x (float): angle in degrees

    Returns:
        float: angle in degrees
    """
    if (x >= -180) and (x < 180):
        return x
    if (x >= -360) and (x < -180):
        return x + 360
    if (x >= 180) and (x < 360):
        return x - 360

    return convert_0_2pi_to_negpi_pi(x % (np.sign(x) * 360))


def convert_negpi_pi_to_0_2pi(x: float) -> float:
    """Convert angle from -pi to pi to 0 to 2pi for an input in degrees"""
    if x >= 0:
        return x % 360
    if (x < 0) & (x >= -360):
        return x + 360

    return convert_negpi_pi_to_0_2pi(x % -360)


def get_neighbor_list(starting_atoms: Atoms) -> np.ndarray:
    """Get the neighbor list from an ase.Atoms object

    Args:
        starting_atoms: ase.Atoms object

    Returns:
        np.ndarray: neighbor list
    """
    from ase.neighborlist import build_neighbor_list

    neighborlist = build_neighbor_list(
        atoms=starting_atoms,
        self_interaction=False,
    )
    conn_matrix = neighborlist.get_connectivity_matrix().todense()
    nbr_list = np.stack(conn_matrix.nonzero(), axis=-1)
    return nbr_list


def find_angles_from_bonds(bonds: np.ndarray) -> np.ndarray:
    """Find indices of atoms involved in all possible angles from a list of bonds/nbr_list

    Args:
        bonds: list of bonds/nbr_list (num_bonds x 2)

    Returns:
        list of angles (num_angles x 3)
    """
    angles = []
    for i in range(bonds.shape[0]):
        for j in range(bonds.shape[0]):
            if i != j:
                # Check if they share an atom
                shared_atom = np.intersect1d(bonds[i], bonds[j])
                if len(shared_atom) == 1:
                    # Sort to ensure the unique representation of an angle
                    angle = np.sort(
                        [
                            bonds[i][bonds[i] != shared_atom[0]][0],
                            shared_atom[0],
                            bonds[j][bonds[j] != shared_atom[0]][0],
                        ]
                    )
                    # Add the angle if it's not a bond and not already added
                    if angle[0] != angle[2] and not any(
                        np.array_equal(angle, existing_angle) for existing_angle in angles
                    ):
                        angles.append(angle)
    return np.array(angles)


def get_bond_distances(
    atoms: Atoms,
    nbr_list: np.array = None,
    periodic: bool = False,
) -> dict:
    """Get all bond distances in an Atoms object

    Args:
        atoms (Atoms): ase.Atoms object
        nbr_list (np.array, optional): the neighbor list for the Atoms. Defaults to None.
        periodic (bool, optional): if true, the structure is periodic. Defaults to False.

    Returns:
        dict: _description_
    """
    dist_mat = atoms.get_all_distances(mic=periodic)

    if nbr_list is None:
        bond_dist = {}
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if j >= i:
                    continue
                bond_dist[(i, j)] = dist_mat[i, j]
    else:
        dist = dist_mat[nbr_list[:, 0], nbr_list[:, 1]]
        bond_dist = {(i[0], i[1]): d for i, d in zip(nbr_list, dist)}

    return bond_dist


def get_bond_angles(atoms: Atoms, angle_inds: np.array) -> dict:
    """Get all bond angles in an Atoms object

    Args:
        atoms (Atoms): ase.Atoms object
        angle_inds (np.array): indices of atoms involved in the angles

    Returns:
        dict: the bond angles with a tuple of atom indices as keys and
            the angle in degrees as values
    """
    angles = atoms.get_angles(angle_inds)
    angles_dict = {(i[0], i[1], i[2]): a for i, a in zip(angle_inds, angles)}

    return angles_dict


def get_bond_dihedrals(
    atoms: Atoms,
    dihedral_inds: np.ndarray,
) -> dict:
    """Get all dihedral angles in an Atoms object

    Args:
        atoms (Atoms): ase.Atoms object
        dihedral_inds (np.ndarray): indices of atoms involved in the dihedrals

    Returns:
        dict: the dihedral angles with a tuple of atom indices as keys and
            the dihedral in degrees as values
    """
    diheds = atoms.get_dihedrals(dihedral_inds)
    dihed_dict = {(i[0], i[1], i[2], i[3]): d for i, d in zip(dihedral_inds, diheds)}

    return dihed_dict


def get_dihedrals(atoms: Atoms, dihedral_inds: np.ndarray) -> list:
    """Get dihedral angles in an Atoms object in the range -pi to pi

    Args:
        atoms (Atoms): ase.Atoms object
        dihedral_inds (np.ndarray): indices of atoms involved in the dihedrals

    Returns:
        list: dihedral angles in degrees
    """
    dihed = atoms.get_dihedrals(dihedral_inds)
    dihed = [convert_0_2pi_to_negpi_pi(i) for i in dihed]
    return dihed


def featurize_bond_info(
    atoms: Atoms,
    nbr_list: np.array | None = None,
    angle_inds: np.array | None = None,
    dihedral_inds: np.array | None = None,
    periodic: bool = False,
) -> dict:
    """Featurize bond information in an Atoms object with optional neighbor list,
    bond angles, and dihedral angles

    Args:
        atoms (Atoms): ase.Atoms object
        nbr_list (np.array, optional): the neighbor list for the Atoms. Defaults to None.
        angle_inds (np.array, optional): atom indices for which to determine bond angles.
            Defaults to None.
        dihedral_inds (np.array, optional): atom indices for which to determine dihedral angles.
            Defaults to None.
        periodic (bool, optional): If true, the structure is periodic. Defaults to False.

    Returns:
        dict: _description_
    """
    bond_info = {}
    # get all bond distances
    bond_info.update(
        get_bond_distances(
            atoms=atoms,
            nbr_list=nbr_list,
            periodic=periodic,
        )
    )

    # get all bond angles
    if angle_inds is not None:
        bond_info.update(
            get_bond_angles(
                atoms=atoms,
                angle_inds=angle_inds,
            )
        )

    # get all dihedral angles
    if dihedral_inds is not None:
        bond_info.update(
            get_bond_dihedrals(
                atoms=atoms,
                dihedral_inds=dihedral_inds,
            )
        )

    return bond_info


def get_atoms(data: dict) -> Atoms:
    """Get an ase.Atoms object from a dictionary

    Args:
        data (dict): dictionary with keys 'nxyz' and 'lattice'
            from which to create an Atoms object

    Returns:
        Atoms: ase.Atoms object
    """
    lattice = data.get("lattice")
    if lattice is not None:
        atoms = Atoms(
            positions=data["nxyz"].numpy()[:, 1:],
            numbers=data["nxyz"].numpy()[:, 0],
            cell=lattice.numpy(),
            pbc=True,
        )
    else:
        atoms = Atoms(
            positions=data["nxyz"].numpy()[:, 1:],
            numbers=data["nxyz"].numpy()[:, 0],
        )

    return atoms


def attach_data_to_atoms(atoms: Atoms, energy: float, forces: list[tuple[float, float, float]], **kwargs) -> Atoms:
    """Attach energy and forces data to an ase.Atoms object

    Args:
        atoms (Atoms): ase.Atoms object
        energy (float): the energy of the Atoms
        forces (list[tuple[float, float, float]]): the forces of the Atoms
        **kwargs: additional data to attach to the Atoms

    Returns:
        Atoms: ase.Atoms object with attached data in the info dictionary (a property of the Atoms)
    """
    _at = atoms.copy()
    _at.set_array("forces", forces)
    _at.info["energy"] = energy

    for key in kwargs:
        _at.info[key] = kwargs[key]

    return _at


def get_atoms_with_data(data: dict) -> Atoms:
    """Get an ase.Atoms object with attached data from a dictionary

    Args:
        data (dict): dictionary with keys 'nxyz', 'energy', and 'energy_grad'
            from which to create an Atoms object with attached data

    Returns:
        Atoms: ase.Atoms object with attached data
    """
    atoms = get_atoms(data)
    identifier = data.get("identifier")
    dihedrals = data.get("dihedrals")
    atoms = attach_data_to_atoms(
        atoms,
        energy=data["energy"].numpy(),
        forces=-data["energy_grad"].numpy(),
        identifier=identifier,
        dihedrals=dihedrals,
    )
    return atoms


def write_atoms_list_to_xyzfile(atoms: Atoms | list[Atoms], filename: str) -> None:
    """Creates an extended XYZ file from an ase Atoms.
    If you want to create a dataset, simply pass a list of atoms
    as `atoms`.

    Args:
        atoms (Atoms | list[Atoms]): ase.Atoms object or list of ase.Atoms objects
        filename (str): path to the output file
    """
    from ase.io import write

    write(filename, atoms, format="extxyz")


def convert_dset_to_atoms_list(dset: list[dict]) -> list[Atoms]:
    """Convert a dataset to a list of ase.Atoms objects

    Args:
        dset (list[dict]): dataset

    Returns:
        list[Atoms]: list of ase.Atoms objects
    """
    xyz = []
    for d in list(dset):
        if "dihedrals" in d:
            d["dihedrals"] = d["dihedrals"].squeeze().numpy()

        at = get_atoms_with_data(d)
        xyz.append(at)

    return xyz
