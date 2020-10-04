"""
Module to deal with statistics of the datasets, removal of outliers
and other statistical functions.
"""

import torch
import numpy as np

from nff.data import Dataset


def remove_outliers(array, std_away=3, max_value=np.inf):
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

    std = np.std(array)
    mean = np.mean(array)

    non_outlier = np.bitwise_and(
        np.abs(array - mean) < std_away * std,
        array < max_value
    )

    non_outlier = np.arange(len(array))[non_outlier]

    return array[non_outlier], non_outlier


def remove_dataset_outliers(dset, reference_key='energy', std_away=3, max_value=np.inf):
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

    _, idx = remove_outliers(array, std_away, max_value)

    new_props = {
        key: [val[i] for i in idx]
        for key, val in dset.props.items()
    }

    return Dataset(new_props, units=dset.units)
