"""
Get fingerprints produced by a CP3D model.
"""

import argparse
import os
import pickle
import numpy as np
import json
import sys
import warnings
import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nff.data import Dataset
from nff.train import load_model
from nff.data import collate_dicts
from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict
from nff.utils import (parse_args, parse_score,
                       CHEMPROP_TRANSFORM, fprint, get_split_names)
from nff.utils.confs import trim_confs

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def get_iter_func(track, num_track=None):
    """
    Get the function to iterate through a process.
    Args:
            track (bool): track this process with tqdm
            num_track (int, optional): number of items
                    that will come out of this process.
    Returns:
            iter_func (callable): iteration function
    """
    if track and num_track != 1:
        iter_func = tqdm
    else:
        def iter_func(x):
            return x
    return iter_func


def save(results,
         targets,
         feat_save_folder,
         prop):
    """
    Save fingerprints, predicted values, true valuse, and various conformer weights.
    Args:
      results (dict): dictionary of the results of applying the CP3D model
      targets (dict): target values
      feat_save_folder (str): folder in which we save the features files
      prop (str): property that you're predicting
    Returns:
      None

    """

    # get the true and predicted values of `prop`

    if prop is None:
        y_true = None
        probas_pred = None

    else:
        probas_pred = (torch.cat(results[prop])
                       .reshape(-1).numpy())

        if prop in targets and targets.get(prop, []):
            y_true = torch.stack(targets[prop]).numpy()

        else:
            y_true = np.ones_like(probas_pred) * float("nan")

    fps = torch.stack(results["fp"]).numpy()
    all_conf_fps = results["conf_fps"]
    learned_weights = results["learned_weights"]
    energy = results.get("energy")
    boltz_weights = results["boltz_weights"]

    smiles_list = targets["smiles"]
    dic = {}

    # whether we're using alpha_ij attention (i.e., every conformer talks
    # to every other), or alpha_i attention (i.e., we just use the conformer's
    # fingerprint to get its weight)

    alpha_ij_att = all([w.reshape(-1).shape[0] == conf_fp.shape[0] ** 2
                        for w, conf_fp in zip(learned_weights, all_conf_fps)])

    for i, smiles in enumerate(smiles_list):

        conf_fps = all_conf_fps[i].numpy()
        these_weights = learned_weights[i].numpy().reshape(-1)
        num_fps = conf_fps.shape[0]

        if alpha_ij_att:
            these_weights = these_weights.reshape(num_fps,
                                                  num_fps)

        dic[smiles] = {"fp": fps[i].reshape(-1),
                       "conf_fps": conf_fps,
                       "learned_weights": these_weights,
                       "boltz_weights": boltz_weights[i].reshape(-1).numpy()}
        if energy is not None:
            dic[smiles].update({"energy": energy[i].reshape(-1).numpy()})

        if y_true is not None and probas_pred is not None:
            dic[smiles].update({"true": y_true[i],
                                "pred": probas_pred[i]})

    with open(feat_save_folder, "wb") as f:
        pickle.dump(dic, f)


def model_from_metric(model, model_folder, metric):
    """
    Get the model with the best validation score according
    to a specified metric.
    Args:
      model (nff.nn.models): original NFF model loaded
      model_folder (str): path to the folder that the model is being trained in
      metric (str): name of metric to use
    Returns:
      model (nff.nn.models): NFF model updated with the state dict of
        the model with the best metric
    """

    # the metric asked for should be in chemprop notation (e.g. auc, prc-auc),
    # but when training a CP3D model we use different names
    # (e.g. roc_auc, prc_auc), so we need to transform into that name

    if metric in CHEMPROP_TRANSFORM:
        use_metric = CHEMPROP_TRANSFORM[metric]

    else:
        use_metric = metric

    # find the best epoch by reading the csv with the metrics
    best_score, best_epoch = parse_score(model_folder, use_metric)
    check_path = os.path.join(model_folder, "checkpoints",
                              f"checkpoint-{best_epoch}.pth.tar")

    state_dict = torch.load(check_path, map_location="cpu"
                            )["model"]
    fprint(f"Loading model state dict from {check_path}")
    model.load_state_dict(state_dict)
    model.eval()

    return model


def fps_and_pred(model, batch, **kwargs):
    """
    Get fingeprints and predictions from the model.
    Args:
      model (nff.nn.models): original NFF model loaded
      batch (dict): batch of data
    Returns:
      results (dict): model predictions and its predicted
        fingerprints, conformer weights, etc.

    """

    model.eval()

    # make the fingerprints
    outputs, xyz = model.make_embeddings(batch, xyz=None, **kwargs)
    # pool to get the learned weights and pooled fingerprints
    pooled_fp, learned_weights = model.pool(outputs)
    # get the final results
    results = model.readout(pooled_fp)
    # add sigmoid if it's a classifier and not in training mode
    if model.classifier:
        keys = list(model.readout.readout.keys())
        for key in keys:
            results[key] = torch.sigmoid(results[key])

    # add any required gradients
    results = model.add_grad(batch=batch, results=results, xyz=xyz)

    # put into a dictionary
    conf_fps = [i.cpu().detach() for i in outputs["conf_fps_by_smiles"]]
    energy = batch.get("energy")
    boltz_weights = batch.get("weights")

    # with operations to de-batch
    n_confs = [(n // m).item()
               for n, m in zip(batch['num_atoms'], batch['mol_size'])]
    for key, val in results.items():
        results[key] = [i for i in val]

    results.update({"fp": [i for i in pooled_fp],
                    "conf_fps": conf_fps,
                    "learned_weights": learned_weights,
                    "boltz_weights": (list(torch.split
                                           (boltz_weights, n_confs)))})

    if energy is not None:
        results.update({"energy": list(torch.split(energy, n_confs))})

    return results


def evaluate(model,
             loader,
             device,
             track,
             **kwargs):
    """
    Evaluate a model on a dataset.
    Args:
      model (nff.nn.models): original NFF model loaded
      loader (torch.utils.data.DataLoader): data loader
      device (Union[str, int]): device on which you run the model
    Returns:
      all_results (dict): dictionary of results
      all_batches (dict): dictionary of ground truth
    """

    model.eval()
    model.to(device)

    all_results = []
    all_batches = []

    iter_func = get_iter_func(track)

    for batch in iter_func(loader):

        batch = batch_to(batch, device)
        results = fps_and_pred(model, batch, **kwargs)

        all_results.append(batch_detach(results))

        # don't overload memory with unnecessary keys
        reduced_batch = {key: val for key, val in batch.items()
                         if key not in ['bond_idx', 'ji_idx', 'kj_idx',
                         'nbr_list', 'bonded_nbr_list']}
        all_batches.append(batch_detach(reduced_batch))

    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches


def get_dset_paths(full_path,
                   train_only,
                   val_only,
                   test_only):
    """
    See where the datasets are located and get their paths.
    Args:
      full_path (str): folder with the data in it
      train_only (bool): only load the training set
      val_only (bool): only load the validation set
      test_only (bool): only load the test set
    Returns:
      paths (list): list of paths for each split. Each split
          gets is own sub-list, which either has a single string
          to the corresponding path, or a set of strings if the data
          is broken up into sub-folders.
      dset_names (list[str]): name of the splits
        (e.g. train, val, test)
    """

    dset_names = get_split_names(train_only=train_only,
                                 val_only=val_only,
                                 test_only=test_only)

    # see if the datasets are in the main folder
    main_folder = all([os.path.isfile(os.path.join(full_path, name
                                                   + ".pth.tar")) for name
                       in dset_names])

    if main_folder:
        paths = [[os.path.join(full_path, name + ".pth.tar")]
                 for name in dset_names]
    else:
        sub_folders = [i for i in os.listdir(full_path) if i.isdigit()]
        sub_folders = sorted(sub_folders, key=lambda x: int(x))
        paths = [[os.path.join(full_path, i, name + ".pth.tar")
                  for i in sub_folders] for name in dset_names]

    return paths, dset_names


def add_dics(base, new, is_first):
    """
    Add a new dictionary to an old dictionary, where the values in each dictionary
    are lists that should be concatenated, and the keys in the new dictionary might
    not be the same as those in the old one.
    Args:
        base (dict): base dictionary to be added to
        new (dict): new dictionary adding on
        is_first (bool): whether this is the first batch we've loaded
    Returns:
        base (dict): updated base dictionary
    """

    for key, val in new.items():
        if is_first:
            base[key] = val
        else:
            if key in base:
                base[key] += val
    if is_first:
        return base

    # any keys that are new to the dictionary despite this not being
    # the first batch added (i.e. they're in this batch but weren't
    # in previous batches)
    extra_keys = [key for key in new.keys() if key not in
                  base.keys()]

    for key in extra_keys:
        dim = len(base["smiles"])
        old_val = [torch.tensor(float("nan"))
                   for _ in range(dim)]
        new_val = old_val + new[key]
        base[key] = new_val

    # same idea for keys that were here before and aren't now
    missing_keys = [key for key in base.keys() if key not in 
                    new.keys()]

    for key in missing_keys:
        dim = len(new["smiles"])
        base[key] += [torch.tensor(float('nan')) for _ in range(dim)]

    return base


def main(dset_folder,
         device,
         model_folder,
         batch_size,
         prop,
         sub_batch_size,
         feat_save_folder,
         metric=None,
         val_only=False,
         train_only=False,
         test_only=False,
         track=True,
         max_confs=None,
         **kwargs):
    """
    Get fingerprints and predictions from the model.
    Args:
      dset_folder (str): folder with the data in it
      device (Union[str, int]): device on which you run the model
      model_folder (str): path to the folder that the model is being trained in
      batch_size (int): how many data points per batch
      prop (str): property to predict
      sub_batch_size (int): how many conformers to put in memory at a time
      feat_save_folder (str): folder in which we're saving teh features
      metric (str): name of metric to use. If not given, this defaults to
        taking the model with the best validation loss.
      train_only (bool): only load the training set
      val_only (bool): only load the validation set
      test_only (bool): only load the test set
      track (bool): Whether to track progress with tqdm
      max_confs (int): Maximum number of conformers to use when evaluating the
          model
    """

    # get the model initially by taken the one saved as "best_model"
    model = load_model(model_folder)
    # update its state_dict with the checkpoint from the epoch with
    # the best metric score

    if metric is None:
        fprint(("WARNING: You have not specified a metric with which "
                "to choose the best model. Defaulting to whichever was "
                "chosen as the best model during training "))
    else:
        fprint(f"Loading model with best validation {metric}")
        model = model_from_metric(model=model,
                                  model_folder=model_folder,
                                  metric=metric)
    model.eval()

    paths, dset_names = get_dset_paths(dset_folder, train_only=train_only,
                                       val_only=val_only, test_only=test_only)

    # go through each dataset, create a loader, evaluate the model,
    # and save the predictions

    iter_func = get_iter_func(track, num_track=len(dset_names))

    for i in iter_func(range(len(dset_names))):
        results = {}
        targets = {}
        j = 0
        for path in tqdm(paths[i]):
            dataset = Dataset.from_file(path)
            if max_confs is not None:
                dataset = trim_confs(dataset=dataset,
                                     num_confs=max_confs,
                                     idx_dic=None,
                                     enum_func=iter_func)

            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=collate_dicts)

            new_results, new_targets = evaluate(model,
                                                loader,
                                                device=device,
                                                sub_batch_size=sub_batch_size,
                                                track=track)

            is_first = (j == 0)
            results = add_dics(base=results,
                               new=new_results,
                               is_first=is_first)

            targets = add_dics(base=targets,
                               new=new_targets,
                               is_first=is_first)
            j += 1

        name = dset_names[i]
        save_name = f"pred_{metric}_{name}.pickle"
        if feat_save_folder is None:
            feat_save_folder = dset_folder
        if not os.path.isdir(feat_save_folder):
            os.makedirs(feat_save_folder)

        pickle_path = os.path.join(feat_save_folder, save_name)

        save(results=results,
             targets=targets,
             feat_save_folder=pickle_path,
             prop=prop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_folder', type=str,
                        help=("Name of the folder with the "
                              "datasets you want to add "
                              "fingerprints to"))
    parser.add_argument('--feat_save_folder', type=str,
                        help="Path to save pickles")
    parser.add_argument('--no_track', action='store_true',
                        help=("Don't track progress with tqmd "))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments."))
    args = parse_args(parser)
    # need to add this explicitly because `parse_args` will only add
    # the keys that are given as options above
    with open(args.config_file, "r") as f:
        config = json.load(f)

    for key, val in config.items():
        setattr(args, key, val)

    main(**args.__dict__)
