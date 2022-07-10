import numpy as np
import torch
import copy

from torch.utils.data.sampler import Sampler, BatchSampler

REINDEX_KEYS = ['atoms_nbr_list', 'nbr_list', 'bonded_nbr_list',
                'angle_list', 'mol_nbrs']
NBR_LIST_KEYS = ['bond_idx', 'kj_idx', 'ji_idx']
MOL_IDX_KEYS = ['atomwise_mol_list', 'directed_nbr_mol_list',
                'undirected_nbr_mol_list']
IGNORE_KEYS = ['rd_mols']

TYPE_KEYS = {
    'atoms_nbr_list': torch.long,
    'nbr_list': torch.long,
    'num_atoms': torch.long,
    'bond_idx': torch.long,
    'bonded_nbr_list': torch.long,
    'angle_list': torch.long,
    'ji_idx': torch.long,
    'kj_idx': torch.long,
}


def collate_dicts(dicts):
    """Collates dictionaries within a single batch. Automatically reindexes
        neighbor lists and periodic boundary conditions to deal with the batch.

    Args:
        dicts (list of dict): each element of the dataset

    Returns:
        batch (dict)
    """

    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        for key in REINDEX_KEYS:
            if key in d:
                d[key] = d[key] + int(n)

    if all(['nbr_list' in d for d in dicts]):
        # same idea, but for quantities whose maximum value is the length of
        # the nbr list in each batch
        cumulative_nbrs = np.cumsum(
            [0] + [len(d['nbr_list']) for d in dicts])[:-1]
        for n, d in zip(cumulative_nbrs, dicts):
            for key in NBR_LIST_KEYS:
                if key in d:
                    d[key] = d[key] + int(n)

    for key in MOL_IDX_KEYS:
        if key not in dicts[0]:
            continue
        for i, d in enumerate(dicts):
            d[key] += i


    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if key in IGNORE_KEYS:
            continue
        if type(val) == str:
            batch[key] = [data[key] for data in dicts]
        elif hasattr(val, 'shape') and len(val.shape) > 0:
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


class ImbalancedDatasetSampler(Sampler):
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


class BalancedFFSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,
                 balance_type=None,
                 weights=None,
                 balance_dict=None,
                 **kwargs):

        from nff.data.sampling import spec_config_zhu_balance

        if weights is not None:
            self.balance_dict = {"weights": weights}

        elif balance_dict is not None:
            self.balance_dict = balance_dict

        else:
            if balance_type == "spec_config_zhu_balance":
                balance_fn = spec_config_zhu_balance
            else:
                raise NotImplementedError

            balance_dict = balance_fn(**kwargs)
            self.balance_dict = balance_dict

        self.data_length = len(self.balance_dict["weights"])

    def __iter__(self):

        return (i for i in torch.multinomial(
            self.balance_dict["weights"],
            self.data_length,
            replacement=True))

    def __len__(self):
        return self.data_length


class BalancedBatchedSpecies(BatchSampler):
    def __init__(self,
                 base_sampler,
                 smiles_list,
                 batch_size,
                 min_geoms=None):
        """
        min_geoms (int, optional): minimum number of geoms
            that a species has to have to be sampled
        """
        from nff.data.sampling import get_spec_dic

        BatchSampler.__init__(self,
                              sampler=base_sampler,
                              batch_size=batch_size,
                              drop_last=False)

        self.spec_dic = get_spec_dic({"smiles": smiles_list})
        self.rev_specs = self.reverse_spec_dic()
        self.exclude_sparse(min_geoms)

    def reverse_spec_dic(self):
        # Reverse the spec dic so it maps props index to species
        # Even though self.balance_dict["smiles"] basically is
        # the same thing, because it has the same ordering as
        # `weights`, it uses the SMILES with stereochemistry,
        # and we want them without

        dic = {}
        for key, val in self.spec_dic.items():
            for idx in val:
                dic[int(idx)] = key
        return dic

    def exclude_sparse(self, min_geoms):
        if min_geoms is None:
            return
        invalid_specs = [key for key, val in self.spec_dic.items()
                         if len(val) < min_geoms]
        if not invalid_specs:
            return
        invalid_idx = torch.cat([self.spec_dic[spec] for spec
                                 in invalid_specs])
        self.sampler.balance_dict["weights"][invalid_idx] = 0

    def __iter__(self):

        sampler_indices = list(iter(self.sampler))
        num_samples = len(self.sampler) // self.batch_size
        batch = []

        for i in range(num_samples):
            seed_idx = sampler_indices[i]
            # find out its species
            smiles = self.rev_specs[int(seed_idx)]
            # get other geoms with the same species
            other_geom_idx = self.spec_dic[smiles]
            # set the weights for any geom outside of `other_geom_idx`
            # to zero
            weights = self.sampler.balance_dict["weights"]
            these_weights = copy.deepcopy(weights)
            zero_weight = torch.ones(len(weights),
                                     dtype=torch.bool)
            zero_weight[other_geom_idx] = False
            these_weights[zero_weight] = 0

            # sample indices from this species only
            # by using `these_weights`
            add_idx = torch.multinomial(
                these_weights,
                self.batch_size,
                replacement=True)

            for idx in add_idx:
                batch.append(idx)

            yield batch
            batch = []

    def __len__(self):
        return len(self.sampler) // self.batch_size
