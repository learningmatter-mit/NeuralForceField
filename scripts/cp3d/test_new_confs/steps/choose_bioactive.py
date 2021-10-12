"""
Given a set of fingerprints for each conformer among a set of
N labeled molecules, find the conformer that leads to the
biggest difference in model scores.
"""
import random
import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

from nff.train.evaluate import evaluate
from nff.data import Dataset


def efficient_shrink_dset(dset, idx):
    new_props = {}
    for key, val in dset.props.items():
        new_val = [val[i] for i in idx]
        if isinstance(val, torch.Tensor):
            new_val = torch.Tensor(val)
        new_props[key] = new_val
    sampled_dset = Dataset(props=new_props,
                           check_props=False,
                           do_copy=False)

    return sampled_dset


def sample_batch(dset,
                 targ_key,
                 batch_size,
                 balanced_sampling=True):

    if balanced_sampling:
        idx = []
        for data_cls in [0, 1]:
            possible_keys = [batch['smiles'] for batch in dset
                             if batch[targ_key] == data_cls]
            possible_keys = list(set(possible_keys))
            sampled_keys = random.sample(possible_keys, batch_size // 2)

            idx += [i for i, batch in enumerate(dset) if
                    batch['smiles'] in sampled_keys]

    else:
        possible_keys = list(set([batch['smiles'] for batch in dset]))
        sampled_keys = random.sample(possible_keys, batch_size)
        idx = [i for i, batch in enumerate(dset) if batch['smiles']
               in sampled_keys]

    sampled_dset = efficient_shrink_dset(dset=dset,
                                         idx=idx)

    return sampled_dset


def make_sample_pairs(sampled_dset,
                      targ_key):

    smiles_by_cls = {}

    for data_cls in [0, 1]:
        smiles_set = [batch['smiles'] for batch in sampled_dset
                      if batch[targ_key] == data_cls]
        smiles_by_cls[data_cls] = list(set(smiles_set))

    sample_pairs = []
    for hit in smiles_by_cls[1]:
        for miss in smiles_by_cls[0]:
            dic = {"hit": hit, "miss": miss}
            sample_pairs.append(dic)

    return sample_pairs


def get_best_conf(sampled_dset,
                  sample_pair,
                  sim_func,
                  fp_key='fp'):

    fps = {"hit": [], "miss": []}
    fp_ids = {"hit": [], "miss": []}

    for batch in sampled_dset:
        smiles = batch['smiles']
        if smiles == sample_pair['hit']:
            key = 'hit'
        elif smiles == sample_pair['miss']:
            key = 'miss'
        else:
            continue

        fps[key].append(np.array(batch[fp_key]))
        fp_ids[key].append(batch['geom_id'])

    fps = {key: np.stack(val) for key, val in fps.items()}

    # n_hit_confs x n_miss_confs array
    sim = sim_func(fps['hit'], fps['miss'])

    # take maximum similarity along the `miss` axis
    max_sim = sim.max(axis=-1)
    # take fp with minimum similarity
    best_hit_idx = max_sim.argmin()
    best_miss_idx = sim.argmax(axis=-1)[best_hit_idx]

    best_hit_id = fp_ids['hit'][best_hit_idx]
    best_miss_id = fp_ids['miss'][best_miss_idx]

    geom_ids = {"hit": best_hit_id, "miss": best_miss_id}

    return geom_ids


def exhaustive_sampling(sampled_dset,
                        targ_key):
    # First make all sets of smiles pairs that are to be
    # maximally separated

    sample_pairs = make_sample_pairs(sampled_dset=sampled_dset,
                                     targ_key=targ_key)

    # Then compute the best conformer FPs from each pair,
    # and add them to a dictionary, which has the form
    # {smiles_1: [fp1_geom_id, fp2_geom_id, ...], smiles_2: {...},
    # etc.}

    fp_ids = {}

    for sample_pair in sample_pairs:
        these_fp_ids = get_best_conf(sampled_dset=sampled_dset,
                                     sample_pair=sample_pair)
        for key, smiles in sample_pairs.items():
            if smiles not in fp_ids:
                fp_ids[smiles] = []

            # key can be `hit` or `miss`
            geom_id = these_fp_ids[key]
            fp_ids[smiles].append(geom_id)

    return fp_ids


def update_fps(all_fp_ids,
               these_fp_ids):

    for key, val in these_fp_ids.items():
        all_fp_ids[key] = all_fp_ids.get(key, []) + val


def get_mode_fps(all_fp_ids):
    final_fps = {}
    for key, fps_lst in all_fp_ids.items():
        best_geom_id = stats.mode(fps_lst).mode.item()
        final_fps[key] = best_geom_id

    return final_fps


def choose_confs(dset,
                 num_batches,
                 targ_key,
                 batch_size,
                 balanced_sampling):

    # First make fingerprints for each conformer

    # Then do exhaustive sampling among batches to
    # generate samples of the best conformers

    all_fp_ids = {}

    for _ in range(num_batches):
        sampled_dset = sample_batch(dset=dset,
                                    targ_key=targ_key,
                                    batch_size=batch_size,
                                    balanced_sampling=balanced_sampling)
        these_fp_ids = exhaustive_sampling(sampled_dset=sampled_dset,
                                           targ_key=targ_key)
        update_fps(all_fp_ids=all_fp_ids,
                   these_fp_ids=these_fp_ids)

    # And finally take the conformer for each smiles that has the most
    # counts

    final_fps = get_mode_fps(all_fp_ids)

    return final_fps


def null_loss(x, y):
    return torch.Tensor([0])


def confs_of_species(all_results,
                     all_batches,
                     targ_key,
                     smiles):

    use_idx = [i for i, batch_smiles in enumerate(all_batches['smiles'])
               if batch_smiles == smiles]

    best_loss = float('inf')
    for idx in use_idx:
        pred = all_results[targ_key][idx]
        targ = all_batches[targ_key][idx]
        loss = abs(pred - targ)
        if loss < best_loss:
            best_loss = loss
            best_idx = idx

    return best_idx


def choose_confs(model,
                 loader,
                 device,
                 targ_key,
                 dset):
    """
    dset has props organized like this:
            {"nxyz": [nxyz_1, nxyz_2, ..., nxyz_N, ...],
             "smiles": ["C#C", "C#C", ..., "C#N", ...],
             "cov_active": [1, 1, ..., 0, ...]}

    """

    all_results, all_batches = evaluate(model=model,
                                        loader=loader,
                                        loss_fn=null_loss,
                                        device=device)

    smiles_set = list(set([batch['smiles'] for batch in dset]))
    conf_idx = []

    for smiles in smiles_set:
        best_idx = confs_of_species(all_results=all_results,
                                    all_batches=all_batches,
                                    targ_key=targ_key,
                                    smiles=smiles)
        conf_idx.append(best_idx)

    new_dset = efficient_shrink_dset(dset=dset,
                                     idx=best_idx)

    return new_dset


def load_summary(base_dir):
    summary_path = os.path.join(base_dir, 'summary.json')

    with open(summary_path, 'rb') as f:
        summary = json.load(f)

    return summary


def load_pickle(base_dir,
                summary,
                smiles):

    pickle_path = os.path.join(base_dir, summary[smiles]['pickle_path'])

    with open(pickle_path, 'rb') as f:
        rd_dic = pickle.load(f)

    return rd_dic


def rd_mol_to_nxyz(rd_mol):
    atoms = rd_mol.GetAtoms()
    conf = rd_mol.GetConformers()[0]

    xyz = conf.GetPositions()
    n = np.array([atom.GetAtomicNum() for atom in atoms])
    nxyz = np.concatenate([n.reshape(-1, 1), xyz], axis=-1)

    return nxyz


def make_dset(base_dir,
              targ_key,
              cutoff,
              undirected):
    """
    Make combined dataset from separate RDKit pickles
    """
    summary = load_summary(base_dir)

    other_keys = ["smiles", targ_key]
    props = {key: [] for key in other_keys}
    props["nxyz"] = []

    for smiles in tqdm(list(summary.keys())):
        rd_dic = load_pickle(base_dir=base_dir,
                             summary=summary,
                             smiles=smiles)
        nxyz_set = [rd_mol_to_nxyz(conf['rd_mol'])
                    for conf in rd_dic['confs']]

        for nxyz in nxyz_set:
            props["nxyz"].append(torch.Tensor(nxyz))
            for key in other_keys:
                props[key].append(rd_dic[key])

    dset = Dataset(props,
                   check_props=True,
                   do_copy=False)

    dset.generate_neighbor_list(cutoff=cutoff,
                                undirected=undirected)

    return dset


def get_dset(rd_mol_dir,
             dset_dir,
             dset_name,
             targ_key,
             cutoff,
             undirected,
             load_existing=True):

    dset_path = os.path.join(dset_dir, dset_name)
    if os.path.isfile(dset_path) and load_existing:
        dset = Dataset.from_file(dset_path)
        return dset

    dset = make_dset(base_dir=rd_mol_dir,
                     targ_key=targ_key,
                     cutoff=cutoff,
                     undirected=undirected)
    dset.save(dset_path)

    return dset
