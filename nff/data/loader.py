import numpy as np
from collections.abc import Iterable
import torch
import pdb
from nff.data.topology import ALL_TOPOLOGY_KEYS, RE_INDEX_TOPOLOGY_KEYS

REINDEX_KEYS = ['atoms_nbr_list', 'nbr_list', *RE_INDEX_TOPOLOGY_KEYS]

TYPE_KEYS = {
    'atoms_nbr_list': torch.long,
    'nbr_list': torch.long,
    'num_atoms': torch.long,
    **{key: torch.long for key in ALL_TOPOLOGY_KEYS}}

def collate_dicts(dicts):
    """Collates dictionaries within a single batch. Automatically reindexes neighbor lists
        and periodic boundary conditions to deal with the batch.

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
                d[key] = d[key] + int(n)

    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        # if key == "degree_vec":
        #     pdb.set_trace()
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

    # adjusting the data types:
    for key, dtype in TYPE_KEYS.items():
        if key in batch:
            batch[key] = batch[key].to(dtype)

    return batch
