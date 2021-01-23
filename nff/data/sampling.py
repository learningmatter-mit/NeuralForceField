"""
Tools for balanced sampling of a dataset
"""

import torch

from nff.train.loss import batch_zhu_p
from nff.utils import constants as const
from nff.utils.misc import cat_props
from nff.data import Dataset
from nff.utils.geom import compute_distances


def get_spec_dic(props):
    """
    Find the indices of geoms in the dataset that correspond
    to each species.
    Args:
        props (dict): dataset properties
    Returns:
        spec_dic (dict): dictionary of the form
                {smiles: idx}, where smiles is the smiles
                of a species without cis/trans indicators,
                and idx are the indices of geoms in that
                species in the dataset.
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


def compute_zhu(props,
                zhu_kwargs):
    """
    Compute the approximate Zhu-Nakamura hopping probabilities for
    each geom in the dataset.
    Args:
            props (dict): dataset properties
            zhu_kwargs (dict): dictionary with information about how
                to calculate the hopping rates.
    Returns:
        zhu_p (torch.Tensor): hopping probabilities
    """

    upper_key = zhu_kwargs["upper_key"]
    lower_key = zhu_kwargs["lower_key"]
    expec_gap_kcal = zhu_kwargs["expec_gap"] * const.AU_TO_KCAL["energy"]

    zhu_p = batch_zhu_p(batch=cat_props(props),
                        upper_key=upper_key,
                        lower_key=lower_key,
                        expec_gap=expec_gap_kcal,
                        gap_shape=None)

    return zhu_p


def balanced_spec_zhu(spec_dic,
                      zhu_p):
    """
    Get the Zhu weights assigned to each geom, such that
    the probability of getting a geom in species A
    is equal to the probability of getting a geom in
    species B [p(A) = p(B)], while the probabiltiies
    within a species are related by p(A, i) / p(A, j)
    = p_zhu(i) / p_zhu(j), where i and j are geometries in
    species A and p_zhu is the Zhu-Nakamura hopping probability.
    Args:
        spec_dic (dict): dictionary with indices of geoms in each
                    species
        zhu_p (torch.Tensor): Zhu-Nakamura hopping probabilities
                    for each geom.
    Returns:
                all_weights (torch.Tensor): sampling weights for each
                                geom in the dataset, normalized to 1.
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


def imbalanced_spec_zhu(zhu_p):
    """
    Get the Zhu weights assigned to each geom, such that
    the probability of getting **any** geom i in the dataset
    is related to the probability of getting **any** geom j
    through p(i) / p(j) = p_zhu(i) / p_zhu(j),
    p_zhu is the Zhu-Nakamura hopping probability. This
    is not balanced with respect to species, so a species
    that has more geoms with high p_zhu will get sampled
    more often.

    Args:
            zhu_p (torch.Tensor): Zhu-Nakamura hopping probabilities
                            for each geom
    Returns:
            all_weights (torch.Tensor): sampling weights for each
                            geom in the dataset, normalized to 1
    """

    all_weights = zhu_p / zhu_p.sum()
    return all_weights


def assign_clusters(ref_idx,
                    spec_nxyz,
                    ref_nxyzs,
                    device):
    """
    Assign each geom to a cluster.

    Args:
        ref_idx (torch.LongTensor): atom indices
            to consider in the RMSD computation between reference
            nxyz and geom nxyz. For example, if you want to associate
            a geometry to a cis or trans cluster, you only really want
            the RMSD of the CNNC atoms with respect to those in the
            converged cis or trans geoms.
        spec_nxyz (list[torch.Tensor]): list of nxyz's for this
            species.
        ref_nxyzs (list[list[torch.Tensor]]): the reference xyz's that
            you want to include in your sampling (e.g. cis,
            trans, and CI). Every xyz will be assigned to the
            one of these three states that it is closest to.
            These three states will then be evenly sampled.
            Note that each state gets its own list of tensors,
            because one state can have more than one geom (e.g. there
            might be multiple distinct CI geoms).
         device (str): device on which to do the RMSD calculations
    Returns:
        cluster_dic (dict): dictionary of the form {cluster: idx},
            where cluster is the cluster number and idx is the set of
            indices of geoms that belong to that cluster.
    """
    # assign a cluster to each nxyz by computing its RMSD with respect
    # to each reference nxyz and selecting the one with the smallest
    # distance

    # we'll make datasets so we can use them as input to the torch
    # parallelized distance computation

    # the first is just the set of geom nxy's
    props_1 = {"nxyz": [i[ref_idx] for i in spec_nxyz]}
    # the second is the reference dataset
    props_0 = {"nxyz": []}

    # use `cluster_idx` to keep track of which reference geoms belong
    # to which cluster, because  one cluster can have many reference
    # geoms

    cluster_idx = {}
    for i, ref_nxyz_lst in enumerate(ref_nxyzs):
        cluster_idx[i] = torch.arange(len(ref_nxyz_lst))
        if i != 0:
            cluster_idx[i] += cluster_idx[i - 1][-1] + 1

        for ref_nxyz in ref_nxyz_lst:
            props_0["nxyz"].append(ref_nxyz[ref_idx])

    # compute the rmsds
    dset_0 = Dataset(props=props_0)
    dset_1 = Dataset(props=props_1)

    rmsds, _ = compute_distances(dataset=dset_0,
                                 device=device,
                                 dataset_1=dset_1)

    # take the minimum rmsd with respect to the set of reference
    # nxyz's in each cluster
    num_clusters = len(ref_nxyzs)
    min_rmsds = torch.zeros(len(spec_nxyz), num_clusters)

    for cluster, idx in cluster_idx.items():
        these_rmsds = rmsds[idx]
        these_mins, _ = these_rmsds.min(0)
        min_rmsds[:, cluster] = these_mins

    # assign a cluster to each species and record in `cluster_dic`
    clusters = min_rmsds.argmin(-1)
    cluster_dic = {i: [] for i in range(num_clusters)}

    for spec_idx, cluster in enumerate(clusters):
        cluster_dic[cluster.item()].append(spec_idx)

    return cluster_dic


def per_spec_config_weights(spec_nxyz,
                            ref_nxyzs,
                            ref_idx,
                            device='cpu'):
    """
    Get weights to evenly sample different regions of phase
    space for a given species
    Args:
        spec_nxyz (list[torch.Tensor]): list of nxyz's for this
            species.
        ref_nxyzs (list[list[torch.Tensor]]): the reference xyz's that
            you want to include in your sampling (e.g. cis,
            trans, and CI). Every xyz will be assigned to the
            one of these three states that it is closest to.
            These three states will then be evenly sampled.
            Note that each state gets its own list of tensors,
            because one state can have more than one geom (e.g. there
            might be multiple distinct CI geoms).
        ref_idx (torch.LongTensor): atom indices
            to consider in the RMSD computation between reference
            nxyz and geom nxyz. For example, if you want to associate
            a geometry to a cis or trans cluster, you only really want
            the RMSD of the CNNC atoms with respect to those in the
            converged cis or trans geoms.
         device (str): device on which to do the RMSD calculations
    Returns:
        geom_weights(torch.Tensor): weights for each geom of this species,
                        normalized to 1.


    """

    # a dictionary that tells you which geoms are in each cluster
    cluster_dic = assign_clusters(ref_idx=ref_idx,
                                  spec_nxyz=spec_nxyz,
                                  ref_nxyzs=ref_nxyzs,
                                  device=device)

    # assign weights to each geom equal to 1 / (num geoms in cluster),
    # so that the probability of sampling any one cluster is equal to
    # the probability of sampling any other

    num_geoms = len(spec_nxyz)
    geom_weights = torch.zeros(num_geoms)

    for idx in cluster_dic.values():
        if len(idx) == 0:
            continue
        geom_weight = 1 / len(idx)
        # print(geom_weight)
        torch_idx = torch.LongTensor(idx)
        geom_weights[torch_idx] = geom_weight

    # return normalized weights
    geom_weights /= geom_weights.sum()

    return geom_weights


def all_spec_config_weights(props,
                            ref_nxyz_dic,
                            spec_dic,
                            device):
    """
    Get the "configuration weights" for each geom, i.e.
    the weights chosen to evenly sample each cluster
    for each species.
    Args:
        props (dict): dataset properties
        ref_nxyz_dic (dict): dictionary of the form
            {smiles: [{"nxyz": ref_nxyz,
            "idx": idx}]}, where smiles is
            the smiles without cis/trans info, the
            ref_nxyzs are the reference nxyz's
            for that species, and idx are the indices
            of the atoms in the RMSD computation
            with respect to the reference.
        spec_dic (dict): dictionary with indices of geoms in each
                species
        device (str): device on which to do the RMSD calculations
    Returns:
        weight_dic(dict): dictionary of the form {smiles: geom_weights},
            where geom_weights are the set of normalized weights for each
            geometry in that species.

    """

    weight_dic = {}
    for spec, idx in spec_dic.items():
        ref_nxyzs = ref_nxyz_dic[spec]['nxyz']
        ref_idx = ref_nxyz_dic[spec]['idx']
        spec_nxyz = [props['nxyz'][i] for i in idx]
        geom_weights = per_spec_config_weights(
            spec_nxyz=spec_nxyz,
            ref_nxyzs=ref_nxyzs,
            ref_idx=ref_idx,
            device=device)
        weight_dic[spec] = geom_weights

    return weight_dic


def balanced_spec_config(weight_dic,
                         spec_dic):
    """
    Generate weights for geoms such that there is balance with respect
    to species [p(A) = p(B)], and with respect to clusters in each
    species [p(A, c1) = p(A, c2), where c1 and c2 are two different
    clusters in species A].
    Args:
        spec_dic (dict): dictionary with indices of geoms in each species.
        weight_dic (dict): dictionary of the form {smiles: geom_weights},
                where geom_weights are the set of normalized weights for each
                geometry in that species.
        Returns:
                all_weights (torch.Tensor): normalized set of weights
    """

    num_geoms = sum([i.shape[0] for i in weight_dic.values()])
    all_weights = torch.zeros(num_geoms)

    for key, idx in spec_dic.items():
        all_weights[idx] = weight_dic[key]

    all_weights /= all_weights.sum()

    return all_weights


def imbalanced_spec_config(weight_dic,
                           spec_dic):
    """
        Generate weights for geoms such that there is no balance with respect
        to species [p(A) != p(B)], but there is with respect to clusters in
        each species [p(A, c1) = p(A, c2), where c1 and c2 are two different
        clusters in species A].
        Args:
            spec_dic (dict): dictionary with indices of geoms in each species.
            weight_dic (dict): dictionary of the form {smiles: geom_weights},
                    where geom_weights are the set of normalized weights for
                    each geometry in that species.
            Returns:
                    all_weights (torch.Tensor): normalized set of weights
    """

    num_geoms = sum([i.shape[0] for i in weight_dic.values()])
    all_weights = torch.zeros(num_geoms)

    for key, idx in spec_dic.items():
        num_spec_geoms = len(idx)
        all_weights[idx] = weight_dic[key] * num_spec_geoms

    all_weights /= all_weights.sum()

    return all_weights


def get_rand_weights(spec_dic):
    """
    Generate weights for random sampling of geoms - i.e., equal weights
    for every geometry.
    Args:
        spec_dic (dict): dictionary with indices of geoms in each species.
    Returns:
        balanced_spec_weights (torch.Tensor): weights generated so that
                P(A) = P(B) for species A and B, and p(A, i) = p(A, j)
                for any geoms within A.
        imbalanced_spec_weights (torch.Tensor): weights generated so that
                P(A) != P(B) in general for species A and B, and p(i) = p(j)
                for any geoms.
    """

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


def combine_weights(balanced_config,
                    imbalanced_config,
                    balanced_zhu,
                    imbalanced_zhu,
                    balanced_rand,
                    imbalanced_rand,
                    spec_weight,
                    config_weight,
                    zhu_weight):
    """
    Combine config weights, Zhu-Nakamura weights, and random
    weights to get the final weights for each geom.
    Args:
        balanced_config (torch.Tensor): config weights with
                species balancing
        imbalanced_config (torch.Tensor): config weights without
                species balancing
        balanced_zhu (torch.Tensor): Zhu weights with
                species balancing
        imbalanced_zhu (torch.Tensor): Zhu weights without
                species balancing
        balanced_rand (torch.Tensor): equal weights with
                species balancing
        imbalanced_rand (torch.Tensor): equal weights without
                species balancing
        spec_weight (float): weight given to equal species balancing.
                If equal to 1, then p(A) = p(B) for all species. If equal
                to 0, species are not considered at all for balancing.
                Intermediate values reflect the extent to which you care
                about balancing species during sampling.
        config_weight (float): the weight given to balance among
                configurations. Must be <= 1.
        zhu_weight (float): the weight given to sampling high-hopping rate.
                geoms. Must be <= 1 and satisfy `config_weight` + `zhu_weight`
                <= 1. Thedifference, 1 - config_weight - zhu_weight, is the
                weight given to purely random sampling.
    Returns:
        final_weights (torch.Tensor): final weights for all geoms, normalized
                to 1.
    """

    # combination of zhu weights that are balanced and imbalanced with respect
    # to species

    weighted_zhu = (balanced_zhu * zhu_weight * spec_weight
                    + imbalanced_zhu * zhu_weight * (1 - spec_weight))

    # combination of config weights that are balanced and imbalanced with
    # respect to species
    weighted_config = (balanced_config * config_weight * spec_weight
                       + imbalanced_config * config_weight * (1 - spec_weight))

    # combination of random weights that are balanced and imbalanced with
    # respect to species

    rand_weight = (1 - zhu_weight - config_weight)
    weighted_rand = (balanced_rand * rand_weight * spec_weight
                     + imbalanced_rand * rand_weight * (1 - spec_weight))

    # final weights

    final_weights = weighted_zhu + weighted_config + weighted_rand

    return final_weights


def spec_config_zhu_balance(props,
                            ref_nxyz_dic,
                            zhu_kwargs,
                            spec_weight,
                            config_weight,
                            zhu_weight,
                            device='cpu'):
    """
    Generate weights that combine balancing of species,
    configurations, and Zhu-Nakamura hopping rates.
    Args:
        props (dict): dataset properties
                zhu_kwargs (dict): dictionary with information about how
                to calculate the hopping rates.
        spec_weight (float): weight given to equal species balancing.
                If equal to 1, then p(A) = p(B) for all species. If equal
                to 0, species are not considered at all for balancing.
                Intermediate values reflect the extent to which you care
                about balancing species during sampling.
        config_weight (float): the weight given to balance among configurations.
                Must be <= 1.
        zhu_weight (float): the weight given to sampling high-hopping rate geoms.
                Must be <= 1 and satisfy `config_weight` + `zhu_weight` <= 1. The
                difference, 1 - config_weight - zhu_weight, is the weight given to
                purely random sampling.
        device (str): device on which to do the RMSD calculations
    Returns:
        final_weights (torch.Tensor): final weights for all geoms,
            normalized to 1.
    """
    spec_dic = get_spec_dic(props)

    # get the species-balanced and species-imbalanced
    # configuration weights

    config_weight_dic = all_spec_config_weights(
        props=props,
        ref_nxyz_dic=ref_nxyz_dic,
        spec_dic=spec_dic,
        device=device)

    balanced_config = balanced_spec_config(
        weight_dic=config_weight_dic,
        spec_dic=spec_dic)

    imbalanced_config = imbalanced_spec_config(
        weight_dic=config_weight_dic,
        spec_dic=spec_dic)

    # get the species-balanced and species-imbalanced
    # zhu weights

    zhu_p = compute_zhu(props=props,
                        zhu_kwargs=zhu_kwargs)
    balanced_zhu = balanced_spec_zhu(spec_dic=spec_dic,
                                     zhu_p=zhu_p)
    imbalanced_zhu = imbalanced_spec_zhu(zhu_p=zhu_p)

    # get the random weights
    balanced_rand, imbalanced_rand = get_rand_weights(
        spec_dic=spec_dic)

    # combine them all together

    final_weights = combine_weights(
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
