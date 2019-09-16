import numpy as np 
from collections.abc import Iterable

import torch 

REINDEX_KEYS = ['nbr_list', 'pbc']

def collate_dicts(dicts):
    """Collates dictionaries within a single batch.

    Args:
        dicts (list of dict): each element of the dataset

    Returns:
        batch (dict)
    """

    # new indices for the batch: the first one is zero and the last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        for key in REINDEX_KEYS:
            if key in d:
                d[key] = d[key] + n

    batch = {}
    for key, val in dicts[0].items():
        if type(val) == str:
            batch[key] = [data[key] for data in dicts]
        elif len(val.shape) > 0:
            batch[key] = torch.cat([
                data[key]
                for data in dicts
            ], dim=0)
        else:
            batch[key] = torch.stack(
                [data[key] for data in dicts],
                dim=0
            )

    return batch

