import numpy as np
import torch
import random

from nff.utils import constants as const
from nff.utils.misc import cat_props

REINDEX_KEYS = ['atoms_nbr_list', 'nbr_list', 'bonded_nbr_list', 'angle_list']
NBR_LIST_KEYS = ['bond_idx', 'kj_idx', 'ji_idx']
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


class BalancedFFSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,
                 props,
                 ref_nxyz_dic,
                 zhu_kwargs,
                 spec_weight,
                 config_weight,
                 zhu_weight):

        self.all_weights = self.get_all_weights(props=props,
                                                ref_nxyz_dic=ref_nxyz_dic,
                                                zhu_kwargs=zhu_kwargs,
                                                spec_weight=spec_weight,
                                                config_weight=config_weight,
                                                zhu_weight=zhu_weight)
        self.data_length = len(self.all_weights)

    def get_spec_dic(self, props):
        """
        Assign weights to evenly sample different species.
        """
        spec_dic = {}
        for i, spec in enumerate(props["smiles"]):
            no_stereo_spec = (spec.replace("\\", "")
                              .replace("/", ""))
            if no_stereo_spec not in spec_dic:
                spec_dic[no_stereo_spec] = []
            spec_dic[no_stereo_spec].append(i)

        for key, val in spec_dic.items():
            spec_dic[key] = torch.LongTensor(val)

        return spec_dic

    def compute_zhu(self,
                    props,
                    zhu_kwargs):

        from nff.train.loss import batch_zhu_p

        upper_key = zhu_kwargs["upper_key"]
        lower_key = zhu_kwargs["lower_key"]
        expec_gap_kcal = zhu_kwargs["expec_gap"] * const.AU_TO_KCAL["energy"]

        zhu_p = batch_zhu_p(batch=cat_props(props),
                            upper_key=upper_key,
                            lower_key=lower_key,
                            expec_gap=expec_gap_kcal,
                            gap_shape=None)

        return zhu_p

    def balanced_spec_zhu(self,
                          spec_dic,
                          zhu_p):
        """
        Get the Zhu weights assigned to each geom, such that
        the probability of getting any geom in species A
        is equal to the probability of getting any geom in
        species B, and p(A, i) / p(A, j) = p_zhu(i) / p_zhu(j),
        where i and j are geometries in species A and p_zhu
        is the Zhu-Nakamura hopping probability.
        """

        num_geoms = sum([len(i) for i in spec_dic.values()])
        all_weights = torch.zeros(num_geoms)

        for lst_idx in spec_dic.values():

            idx = torch.LongTensor(lst_idx)
            this_zhu = zhu_p[idx]
            sum_zhu = this_zhu.sum()

            # make sure they're normalized first
            if sum_zhu != 0:
                this_zhu /= sum_zhu

            all_weights[idx] = this_zhu

        all_weights /= all_weights.sum()

        return all_weights

    def imbalanced_spec_zhu(self, zhu_p):
        """
        Get the Zhu weights assigned to each geom, such that
        the probability of getting any geom i in the dataset
        is related to the probability of getting any geom j
        through p(i) / p(j) = p_zhu(i) / p_zhu(j),
        p_zhu is the Zhu-Nakamura hopping probability. This
        is not balanced with respect to species, so a species
        that has more geoms with high p_zhu will get sampled
        more often.
        """

        all_weights = zhu_p / zhu_p.sum()
        return all_weights

    def per_spec_config_weights(self,
                                spec_nxyz,
                                ref_nxyzs):
        """
        Get weights to evenly sample different regions of phase
        space for a given species
        Args:
            props (dict): props
            ref_nxyzs (list[torch.Tensor]): the reference xyz's that
                you want to include in your sampling (e.g. cis,
                trans, and CI). Every xyz will be assigned to the one
                of these three states that it is closest to. These
                three states will then be evenly sampled.

        """

        # a list of the cluster that each geom belongs to

        # We need to generalize this for multiple species and probably
        # do either (a) a clustering algorithm, or (b) a division of
        # distances between the two reference structures

        from nff.utils.geom import compute_rmsd

        num_clusters = len(ref_nxyzs)
        cluster_dic = {i: [] for i in range(num_clusters)}

        for i, nxyz in enumerate(spec_nxyz):
            rmsds = [compute_rmsd(targ_nxyz=ref_nxyz, this_nxyz=nxyz)
                     for ref_nxyz in ref_nxyzs]
            cluster = np.argmin(rmsds)
            cluster_dic[cluster].append(i)

        num_geoms = len(spec_nxyz)
        geom_weights = torch.zeros(num_geoms)

        for idx in cluster_dic.values():
            if len(idx) == 0:
                continue
            geom_weight = 1 / len(idx)
            torch_idx = torch.LongTensor(idx)
            geom_weights[torch_idx] = geom_weight

        # return normalized weights
        geom_weights /= geom_weights.sum()

        return geom_weights

    def all_spec_config_weights(self,
                                props,
                                ref_nxyz_dic,
                                spec_dic):
        weight_dic = {}
        for spec, idx in spec_dic.items():
            ref_nxyzs = ref_nxyz_dic[spec]
            spec_nxyz = [props['nxyz'][i] for i
                         in idx]
            geom_weights = self.per_spec_config_weights(
                spec_nxyz=spec_nxyz,
                ref_nxyzs=ref_nxyzs)
            weight_dic[spec] = geom_weights

        return weight_dic

    def balanced_spec_config(self,
                             weight_dic,
                             spec_dic):

        num_geoms = sum([i.shape[0] for i in weight_dic.values()])
        all_weights = torch.zeros(num_geoms)

        for key, idx in spec_dic.items():
            all_weights[idx] = weight_dic[key]

        all_weights /= all_weights.sum()

        return all_weights

    def imbalanced_spec_config(self,
                               weight_dic,
                               spec_dic):

        num_geoms = sum([i.shape[0] for i in weight_dic.values()])
        all_weights = torch.zeros(num_geoms)

        for key, idx in spec_dic.items():
            num_spec_geoms = len(idx)
            all_weights[idx] = weight_dic[key] * num_spec_geoms

        all_weights /= all_weights.sum()

        return all_weights

    def get_rand_weights(self,
                         spec_dic):

        # import pdb
        # pdb.set_trace()

        num_geoms = sum([len(i) for i in spec_dic.values()])

        imbalanced_spec_weights = torch.ones(num_geoms)
        imbalanced_spec_weights /= imbalanced_spec_weights.sum()

        balanced_spec_weights = torch.zeros(num_geoms)
        for idx in spec_dic.values():
            if len(idx) == 0:
                continue
            balanced_spec_weights[idx] = 1 / len(idx)

        total = balanced_spec_weights.sum()
        if total != 0:
            balanced_spec_weights /= total

        return balanced_spec_weights, imbalanced_spec_weights

    def combine_weights(self,
                        balanced_config,
                        imbalanced_config,
                        balanced_zhu,
                        imbalanced_zhu,
                        balanced_rand,
                        imbalanced_rand,
                        spec_weight,
                        config_weight,
                        zhu_weight):
        """
        Notes: 
        (a) `config_weight` + `zhu_weight` <= 1
        - If equal to 1, then all geometries are sampled
        according to their configuration group and zhu weight,
        and possibly their species
        - If not equal to 1, then the remainder get sampled randomly
        (b) spec_weight <= 1 is the weight assigned to evenly sampling
        the species.
        """

        weighted_zhu = (balanced_zhu * zhu_weight * spec_weight
                        + imbalanced_zhu * zhu_weight * (1 - spec_weight))
        weighted_config = (balanced_config * config_weight * spec_weight
                           + imbalanced_config * config_weight
                           * (1 - spec_weight))

        rand_weight = (1 - zhu_weight - config_weight)
        weighted_rand = (balanced_rand * rand_weight * spec_weight
                         + imbalanced_rand * rand_weight * (1 - spec_weight))

        final_weights = weighted_zhu + weighted_config + weighted_rand

        return final_weights

    def get_all_weights(self,
                        props,
                        ref_nxyz_dic,
                        zhu_kwargs,
                        spec_weight,
                        config_weight,
                        zhu_weight):

        spec_dic = self.get_spec_dic(props)

        # get the species-balanced and species-imbalanced
        # configuration weights

        config_weight_dic = self.all_spec_config_weights(
            props=props,
            ref_nxyz_dic=ref_nxyz_dic,
            spec_dic=spec_dic)
        balanced_config = self.balanced_spec_config(
            weight_dic=config_weight_dic,
            spec_dic=spec_dic)
        imbalanced_config = self.imbalanced_spec_config(
            weight_dic=config_weight_dic,
            spec_dic=spec_dic)

        # get the species-balanced and species-imbalanced
        # zhu weights

        zhu_p = self.compute_zhu(props=props,
                                 zhu_kwargs=zhu_kwargs)
        balanced_zhu = self.balanced_spec_zhu(spec_dic=spec_dic,
                                              zhu_p=zhu_p)
        imbalanced_zhu = self.imbalanced_spec_zhu(zhu_p=zhu_p)

        # get the random weights
        balanced_rand, imbalanced_rand = self.get_rand_weights(
            spec_dic=spec_dic)

        # combine them all together

        final_weights = self.combine_weights(
            balanced_config=balanced_config,
            imbalanced_config=imbalanced_config,
            balanced_zhu=balanced_zhu,
            imbalanced_zhu=imbalanced_zhu,
            balanced_rand=balanced_rand,
            imbalanced_rand=imbalanced_rand,
            spec_weight=spec_weight,
            config_weight=config_weight,
            zhu_weight=zhu_weight)

        return final_weights

    def __iter__(self):

        return (i for i in torch.multinomial(
            self.all_weights, self.data_length, replacement=True))

    def __len__(self):
        return self.data_length
