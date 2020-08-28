import numpy as np
import torch
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset as TorchDataset
from operator import itemgetter


from nff.data.topology import ALL_TOPOLOGY_KEYS, RE_INDEX_TOPOLOGY_KEYS


REINDEX_KEYS = ['atoms_nbr_list', 'nbr_list', *RE_INDEX_TOPOLOGY_KEYS]
NBR_LIST_KEYS = ['bond_idx']


TYPE_KEYS = {
    'atoms_nbr_list': torch.long,
    'nbr_list': torch.long,
    'num_atoms': torch.long,
    'bond_idx': torch.long,
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

    if all(['nbr_list' in d for d in dicts]):
        # same idea, but for quantities whose maximum value is the length of the nbr
        # list in each batch
        cumulative_nbrs = np.cumsum([0] + [len(d['nbr_list']) for d in dicts])[:-1]
        for n, d in zip(cumulative_nbrs, dicts):
            for key in NBR_LIST_KEYS:
                if key in d:
                    d[key] = d[key] + int(n)

    # batching the data
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

    # adjusting the data types:
    for key, dtype in TYPE_KEYS.items():
        if key in batch:
            batch[key] = batch[key].to(dtype)

    return batch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """

    Source: https://github.com/ufoym/imbalanced-dataset-sampler/
            blob/master/torchsampler/imbalanced.py

    Sampling class to make sure positive and negative labels
    are represented equally during training.
    Attributes:
        data_length (int): length of dataset
        weights (torch.Tensor): weights of each index in the
            dataset depending.

    """

    def __init__(self,
                 target_name,
                 props):
        """
        Args:
            target_name (str): name of the property being classified
            props (dict): property dictionary
        """


        data_length = len(props[target_name])

        negative_idx = [i for i, target in enumerate(
            props[target_name]) if round(target.item()) == 0]
        positive_idx = [i for i in range(data_length)
                        if i not in negative_idx]

        num_neg = len(negative_idx)
        num_pos = len(positive_idx)

        if num_neg == 0:
            num_neg = 1
        if num_pos == 0:
            num_pos = 1

        negative_weight = num_neg
        positive_weight = num_pos

        self.data_length = data_length
        self.weights = torch.zeros(data_length)
        self.weights[negative_idx] = 1 / negative_weight
        self.weights[positive_idx] = 1 / positive_weight

    def __iter__(self):

        return (i for i in torch.multinomial(
            self.weights, self.data_length, replacement=True))

    def __len__(self):
        return self.data_length



class DatasetFromSampler(TorchDataset):
    """

    Source: https://github.com/catalyst-team/catalyst/blob/
            f98c351dc8b35851040c8477d61ac633d6dd46c3/catalyst/
            data/dataset.py

    Dataset of indexes from `Sampler`."""

    def __init__(self, sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """

    Source: https://github.com/catalyst-team/catalyst/
            blob/master/catalyst/data/sampler.py

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):

        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

