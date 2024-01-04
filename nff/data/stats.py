"""
Module to deal with statistics of the datasets, removal of outliers
and other statistical functions.
"""

import logging

import numpy as np
import torch
from nff.data import Dataset

logger = logging.getLogger(__name__)


def remove_outliers(array, std_away=3, reference_mean=None, reference_std=None, max_value=np.inf):
    """
    Remove outliers from given array using both a number of standard
        deviations and a hard cutoff.

    Args:
        array (np.array): array from which the outliers will be removed.
        std_away (float): maximum number of standard deviations to consider
            a value as outlier.
        max_value (float): cutoff for the values of array. Values higher than
            this cutoff will be considered outliers and thus removed from the
            array.

    Returns:
        array without outliers (np.array)
        non_outlier (np.array): array containing the indices of non-outlier
            values.
    """
    if not reference_mean:
        mean = np.mean(array)
    else:
        mean = reference_mean
    if not reference_std:
        std = np.std(array)
    else:
        std = reference_std

    non_outlier = np.bitwise_and(
        np.abs(array - mean) < std_away * std,
        array < max_value
    )

    non_outlier = np.arange(len(array))[non_outlier]

    return array[non_outlier], non_outlier, mean, std

def remove_dataset_outliers(dset, reference_key='energy', reference_mean=None, reference_std=None, std_away=3, max_value=np.inf):
    """
    Remove outliers from given dataset using both a number of standard
        deviations and a hard cutoff.

    Args:
        dset (nff.data.Dataset): dataset from which the outliers will be removed.
        reference_key (str): key of the dataset which should serve as reference
            when removing the outliers.
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

    _, idx, mean, std = remove_outliers(array, std_away=std_away, reference_mean=reference_mean, reference_std=reference_std, max_value=max_value)

    new_props = {
        key: [val[i] for i in idx]
        for key, val in dset.props.items()
    }
    logging.info(f"reference_mean: {mean}")
    logging.info(f"reference_std: {std}")

    return Dataset(new_props, units=dset.units), mean, std

def center_dataset(dset, reference_key='energy', reference_value=None):
    """
    Center a dataset by subtracting the mean of the reference key.

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


# TODO add function for subtracting the mean of the energies
# check out what CHGNet does


def get_atom_count(formula):
    '''The function `get_atom_count` takes a chemical formula as input and returns a dictionary containing
    the count of each atom in the formula.
    
    Parameters
    ----------
    formula
        The formula parameter is a string representing a chemical formula.
    
    Returns
    -------
        a dictionary containing the count of each atom in the given chemical formula.
    
    '''
    # replace with ase.formula.Formula.count()
    # import re
    # dictio = dict(re.findall('([A-Z][a-z]?)([0-9]*)', formula))
    # for key, val in dictio.items():
    #     dictio[key] = int(val) if val.isdigit() else 1

    # return dictionary
    from ase.formula import Formula
    formula = Formula(formula)
    return formula.count()


def all_atoms(unique_formulas):
    atom_set = set()
    for formula in unique_formulas:
        dictio = get_atom_count(formula)
        atom_set.update(set(dictio.keys()))
    logging.info("atom_set: %s", atom_set)
    return atom_set

def reg_atom_count(formula, atoms):
    '''The function `reg_atom_count` takes a chemical formula and a list of atoms as input, and returns an
    array containing the count of each atom in the formula.
    
    Parameters
    ----------
    formula
        The formula parameter is a string that represents a chemical formula. It can contain elements and
    their corresponding subscripts. For example, "H2O" represents water, where "H" is the element
    hydrogen and "O" is the element oxygen. The subscript "2" indicates that there are two
    atoms
        The `atoms` parameter is a list of strings representing the atoms for which you want to count the
    occurrences in the `formula`.
    
    Returns
    -------
        an array containing the count of each atom in the given formula.
    
    '''
    dictio = get_atom_count(formula)
    count_array = np.array([dictio.get(atom, 0) for atom in atoms])

    return count_array

def get_stoich_dict(dset, formula_key='formula', energy_key='energy'):
    """ The function `get_stoich_dict` takes a dataset, and optional keys for formula and energy, and
    returns a stoichiometry dictionary.
    
    Parameters
    ----------
    dset
        The `dset` parameter is a dataset object that contains properties for each data point. It is
    assumed to have a property for the chemical formula of each data point and a property for the energy
    value of each data point.
    formula_key, optional
        The parameter `formula_key` is a string that represents the key in the dataset properties
    dictionary where the formulas are stored. This key is used to access the formulas for each entry in
    the dataset.
    energy_key, optional
        The `energy_key` parameter is a string that represents the key in the dataset properties where the
    energy values are stored.
    
    """
    # calculates the linear regresion and return the stoich dictionary
    from sklearn import linear_model

    formulas = dset.props[formula_key]
    energies = dset.props[energy_key]
    logging.debug("formulas: %s", formulas)
    logging.debug("energies: %s", energies)

    unique_formulas = list(set(formulas))
    logging.debug("unique formulas: %s", unique_formulas)
    # find the ground state energy for each formula/stoichiometry
    ground_en = [min([energies[i] for i in range(len(formulas)) if formulas[i] == formula]) for formula in unique_formulas]
    unique_atoms = all_atoms(unique_formulas)

    logging.debug("ground_en: %s", ground_en)
    logging.debug("unique atoms: %s", unique_atoms)

    x_in = np.stack([reg_atom_count(formula, unique_atoms)
                        for formula in unique_formulas])

    y_out = np.array(ground_en)

    logging.debug("x_in: %s", x_in)
    logging.debug("y_out: %s", y_out)

    clf = linear_model.LinearRegression()
    clf.fit(x_in, y_out)

    pred = (clf.coef_ * x_in).sum(-1) + clf.intercept_
    # pred = clf.predict(x_in)
    logging.info(f"coef: {clf.coef_}")
    logging.info(f"intercept: {clf.intercept_}")
    logging.debug("pred: %s", pred)
    err = abs(pred - y_out).mean() # in kcal/mol
    logging.info(f"MAE between target energy and stoich energy is {err:.3f} kcal/mol") 
    logging.info("R : %s", clf.score(x_in, y_out))
    fit_dic = {atom: coef for atom, coef in zip(
        unique_atoms, clf.coef_.reshape(-1))}
    stoich_dict = {**fit_dic, "offset": clf.intercept_.item()}
    logging.info(stoich_dict)

    return stoich_dict

def perform_energy_offset(dset, stoic_dict, formula_key='formula', energy_key='energy'):
    '''The function `perform_energy_offset` takes a dataset, a stoichiometry dictionary, and optional keys
    for formula and energy, and performs an energy offset calculation on the dataset.
     
    Parameters
    ----------
    dset
        The `dset` parameter is a dataset object that contains properties for each data point. It is
    assumed to have a property for the chemical formula of each data point and a property for the energy
    value of each data point.
    stoic_dict
        The `stoic_dict` parameter is a dictionary that contains the stoichiometric coefficients for each
    element in the formula. The keys of the dictionary are the element symbols, and the values are the
    corresponding stoichiometric coefficients.
    formula_key, optional
        The parameter `formula_key` is a string that represents the key in the dataset properties
    dictionary where the formulas are stored. This key is used to access the formulas for each entry in
    the dataset.
    energy_key, optional
        The `energy_key` parameter is a string that represents the key in the dataset properties where the
    energy values are stored.
    
    Returns
    -------
        a new dataset (`new_dset`).
    
    '''
    # perform the energy offset
    formulas = dset.props[formula_key]
    energies = dset.props[energy_key]

    if isinstance(energies, torch.Tensor):
        new_energies = energies.clone()
    else:
        new_energies = energies.copy()

    for i, formula in enumerate(formulas):
        dictio = get_atom_count(formula)
        ref_en = 0
        for ele, num in dictio.items():
            ref_en += num * stoic_dict[ele]
        ref_en += stoic_dict['offset']
    
        new_energies[i] -= ref_en
    
    logging.info(f"new_energies: {new_energies}")
    new_dset = dset.copy()
    logging.info(f"old energies: {new_dset.props[energy_key]}")
    new_dset.props[energy_key] = new_energies
    logging.info(f"new energies: {new_dset.props[energy_key]}")
    return new_dset
