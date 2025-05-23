"""Module to deal with statistics of the datasets, removal of outliers and other statistical functions."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase.formula import Formula
from sklearn import linear_model

from nff.data import Dataset

logger = logging.getLogger(__name__)


def remove_outliers(
    array: Union[List, np.ndarray, torch.Tensor],
    std_away: float = 3.0,
    reference_mean: Optional[float] = None,
    reference_std: Optional[float] = None,
    max_value: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Remove outliers from given array using both a number of standard
        deviations and a hard cutoff.

    Args:
        array (Union[List, np.ndarray, torch.Tensor]): array from which the
            outliers will be removed.
        std_away (float): maximum number of standard deviations to consider
            a value as outlier.
        reference_mean (float): mean of the array. If None, the mean of the
            array will be calculated.
        reference_std (float): standard deviation of the array. If None, the
            standard deviation of the array will be calculated.
        max_value (float): cutoff for the values of array. Values higher than
            this cutoff will be considered outliers and thus removed from the
            array.

    Returns:
        array without outliers (np.array)
        non_outlier (np.array): array containing the indices of non-outlier
            values.
        mean (float): mean of the array.
        std (float): standard deviation of the array.
    """

    if isinstance(array, list):
        stats_array = torch.cat(array, dim=0).flatten().cpu().numpy()
        # take the maximum absolute value in the list of tensors
        max_idx = [torch.argmax(torch.abs(ten.flatten())) for ten in array]
        max_values = np.array([array[i].flatten()[max_idx[i]].cpu().numpy() for i in range(len(array))])
    else:
        stats_array = array.copy()
        max_values = stats_array.copy()  # used for outlier removal

    mean = reference_mean if reference_mean else np.mean(stats_array)
    std = reference_std if reference_std else np.std(stats_array)
    non_outlier = np.bitwise_and(np.abs(max_values - mean) < std_away * std, max_values < max_value)

    non_outlier = np.arange(len(array))[non_outlier]
    logging.info("removed %d outliers", len(array) - len(non_outlier))

    if isinstance(array, list):
        filtered_array = [array[i] for i in non_outlier]
        return filtered_array, non_outlier, mean, std

    return array[non_outlier], non_outlier, mean, std


def remove_dataset_outliers(
    dset: Dataset,
    reference_key: str = "energy",
    reference_mean: Optional[float] = None,
    reference_std: Optional[float] = None,
    std_away: float = 3.0,
    max_value: float = np.inf,
) -> Tuple[Dataset, float, float]:
    """Remove outliers from given dataset using both a number of standard
        deviations and a hard cutoff.

    Args:
        dset (nff.data.Dataset): dataset from which the outliers will be removed.
        reference_key (str): key of the dataset which should serve as reference
            when removing the outliers.
        reference_mean (float): mean of the array. If None, the mean of the
            referenced array will be calculated.
        reference_std (float): standard deviation of the array. If None, the
            standard deviation of the referenced array will be calculated.
        std_away (float): maximum number of standard deviations to consider
            a value as outlier.
        max_value (float): cutoff for the values of array. Values higher than
            this cutoff will be considered outliers and thus removed from the
            array.

    Returns:
        new_dset (nff.data.Dataset): new dataset with the bad data removed.
    """
    array = dset.props[reference_key]
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()

    _, idx, mean, std = remove_outliers(
        array,
        std_away=std_away,
        reference_mean=reference_mean,
        reference_std=reference_std,
        max_value=max_value,
    )
    new_props = {key: [val[i] for i in idx] for key, val in dset.props.items()}
    logging.info("reference_mean: %s", mean)
    logging.info("reference_std: %s", std)
    return Dataset(new_props, units=dset.units), mean, std


def center_dataset(
    dset: Dataset, reference_key: str = "energy", reference_value: Optional[float] = None
) -> Tuple[Dataset, float]:
    """Center a dataset by subtracting the mean of the reference key.

    Args:
        dset (nff.data.Dataset): dataset to be centered.
        reference_key (str): key of the dataset which should serve as reference
            when centering the dataset.
        reference_value (float): value of the reference key to be used as
            reference when centering the dataset. If None, the mean of the
            reference key will be used.

    Returns:
        new_dset (nff.data.Dataset): new dataset centered.
    """
    array = dset.props[reference_key]
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()

    if not reference_value:
        reference_value = np.mean(array)
    logging.info("reference_value: %s", reference_value)

    new_dset = dset.copy()
    new_dset.props[reference_key] -= reference_value

    return new_dset, reference_value


def get_atom_count(formula: str) -> Dict[str, int]:
    """Count the number of each atom type in the formula.

    Args:
        formula (str): A chemical formula.

    Returns:
        Dict[str, int]: A dictionary containing the count of each atom in the given
            chemical formula.
    """
    formula = Formula(formula)
    return formula.count()


def all_atoms(unique_formulas: List[str]) -> set:
    """Return a set of all atoms present in the list of formulas.

    Args:
        unique_formulas (List[str]): A list of chemical formula strings.

    Returns:
        set: A set containing all unique atom types found in the provided formulas.
    """
    atom_set = set()
    for formula in unique_formulas:
        dictio = get_atom_count(formula)
        atom_set.update(set(dictio.keys()))
    logging.info("atom_set: %s", atom_set)
    return atom_set


def reg_atom_count(formula: str, atoms: List[str]) -> np.ndarray:
    """Count the occurrence of specified atom types in the formula.

    Args:
        formula (str): A chemical formula string.
        atoms (List[str]): A list of atom types to count in the formula.

    Returns:
        np.ndarray: An array containing the count of each specified atom in the formula.
    """
    dictio = get_atom_count(formula)
    count_array = np.array([dictio.get(atom, 0) for atom in atoms])
    return count_array


def get_stoich_dict(dset: Dataset, formula_key: str = "formula", energy_key: str = "energy") -> Dict[str, float]:
    """Determine per-atom energy coefficients via linear regression.

    Args:
        dset (Dataset): Dataset object containing chemical formulas and energy values.
        formula_key (str, optional): Key for chemical formulas in the dataset properties.
        energy_key (str, optional): Key for energy values in the dataset properties.

    Returns:
        Dict[str, float]: A dictionary with stoichiometric energy coefficients for
            each element, including an 'offset' representing the intercept.
    """
    formulas = dset.props[formula_key]
    energies = dset.props[energy_key]
    logging.debug("formulas: %s", formulas)
    logging.debug("energies: %s", energies)

    unique_formulas = list(set(formulas))
    logging.debug("unique formulas: %s", unique_formulas)
    # Find the ground state energy for each formula/stoichiometry.
    ground_en = [
        min([energies[i] for i in range(len(formulas)) if formulas[i] == formula]) for formula in unique_formulas
    ]
    unique_atoms = all_atoms(unique_formulas)

    logging.debug("ground_en: %s", ground_en)
    logging.debug("unique atoms: %s", unique_atoms)

    x_in = np.stack([reg_atom_count(formula, unique_atoms) for formula in unique_formulas])
    y_out = np.array(ground_en)

    logging.debug("x_in: %s", x_in)
    logging.debug("y_out: %s", y_out)

    clf = linear_model.LinearRegression()
    clf.fit(x_in, y_out)

    pred = (clf.coef_ * x_in).sum(-1) + clf.intercept_
    logging.info("coef: %s", clf.coef_)
    logging.info("intercept: %s", clf.intercept_)
    logging.debug("pred: %s", pred)
    err = abs(pred - y_out).mean()  # in kcal/mol
    logging.info("MAE between target energy and stoich energy is %.3f kcal/mol", err)
    logging.info("R : %s", clf.score(x_in, y_out))
    fit_dic = dict(zip(unique_atoms, clf.coef_.reshape(-1)))
    stoich_dict = {**fit_dic, "offset": clf.intercept_.item()}
    logging.info(stoich_dict)

    return stoich_dict


def perform_energy_offset(
    dset: Dataset,
    stoic_dict: Dict[str, float],
    formula_key: str = "formula",
    energy_key: str = "energy",
) -> Dataset:
    """Perform energy offset calculation by subtracting the reference energy per atom.

    Args:
        dset (Dataset): Dataset object containing chemical formulas and energy values.
        stoic_dict (Dict[str, float]): Dictionary with stoichiometric energy coefficients for
            each element.
        formula_key (str, optional): Key for chemical formulas in the dataset properties.
        energy_key (str, optional): Key for energy values in the dataset properties.

    Returns:
        Dataset: A new dataset with the energy offset applied to each energy value.
    """
    formulas = dset.props[formula_key]
    energies = dset.props[energy_key]

    new_energies = energies.clone() if isinstance(energies, torch.Tensor) else energies.copy()

    for i, formula in enumerate(formulas):
        dictio = get_atom_count(formula)
        ref_en = 0
        for ele, num in dictio.items():
            ref_en += num * stoic_dict[ele]
        ref_en += stoic_dict["offset"]

        new_energies[i] -= ref_en

    logging.info("new_energies: %s", new_energies)
    new_dset = dset.copy()
    logging.info("old energies: %s", new_dset.props[energy_key])
    new_dset.props[energy_key] = new_energies
    logging.info("new energies: %s", new_dset.props[energy_key])
    return new_dset
