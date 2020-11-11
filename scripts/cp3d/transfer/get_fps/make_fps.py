"""
Get fingerprints produced by a CP3D model.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import numpy as np

from nff.data import Dataset
from nff.train import load_model
from nff.data import collate_dicts
from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict
from nff.utils import (tqdm_enum, parse_args, fprint, parse_score,
                       METRICS, CHEMPROP_TRANSFORM)


# dictionary that transforms our metric syntax to chemprop's
REVERSE_TRANSFORM = {val: key for key, val in CHEMPROP_TRANSFORM.items()}
# available metrics
METRIC_LIST = [REVERSE_TRANSFORM.get(metric, metric) for metric in METRICS]

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

        if prop in targets:
            y_true = torch.stack(targets[prop]).numpy()

        else:
            y_true = np.ones_like(probas_pred) * float("nan")

    fps = torch.stack(results["fp"]).numpy()
    all_conf_fps = results["conf_fps"]
    learned_weights = results["learned_weights"]
    energy = results["energy"]
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
                       "energy": energy[i].reshape(-1).numpy(),
                       "boltz_weights": boltz_weights[i].reshape(-1).numpy()}

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

    # load the state dict from the checkpoint of that epoch
    check_path = os.path.join(model_folder, "checkpoints",
                              f"checkpoint-{best_epoch}.pth.tar")

    state_dict = torch.load(check_path, map_location="cpu"
                            )["model"]
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
    n_confs = [(n // m).item() for n, m in zip(batch['num_atoms'], batch['mol_size'])]
    for key, val in results.items():
        results[key] = [i for i in val]
    results.update({"fp": [i for i in pooled_fp],
                    "conf_fps": conf_fps,
                    "learned_weights": learned_weights,
                    "energy": list(torch.split(energy, n_confs)),
                    "boltz_weights": list(torch.split(boltz_weights, n_confs))})
    return results


def evaluate(model,
             loader,
             device,
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

    for i, batch in tqdm_enum(loader):

        batch = batch_to(batch, device)
        results = fps_and_pred(model, batch, **kwargs)

        all_results.append(batch_detach(results))
        all_batches.append(batch_detach(batch))

    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches


def get_dset_paths(full_path, test_only):
    """
    See where the datasets are located and get their paths.
    Args:
      full_path (str): folder with the data in it
      test_only (bool): only load the test set
    Returns:
      paths (list): list of paths for each split. Each split
          gets is own sub-list, which either has a single string
          to the corresponding path, or a set of strings if the data
          is broken up into sub-folders.
      dset_names (list[str]): name of the splits
        (e.g. train, val, test)
    """

    if test_only:
        dset_names = ['test']
    else:
        dset_names = ["train", "val", "test"]

    # see if the datasets are in the main folder
    main_folder = all([os.path.isfile(os.path.join(full_path, name + ".pth.tar"))
                      for name in dset_names])

    if main_folder:
        paths = [[os.path.join(full_path, name + ".pth.tar")] for name in dset_names]
    else:
        sub_folders = [i for i in os.listdir(full_path) if i.isdigit()]
        sub_folders = sorted(sub_folders, key=lambda x: int(x))
        paths = [[os.path.join(full_path, i, name + ".pth.tar") for i in sub_folders]
                 for name in dset_names]

    return paths, dset_names

def add_dics(base, new):
    """
    Add a new dictionary to an old dictionary, where the values in each dictionary
    are lists that should be concatenated, and the keys in the new dictionary might
    not be the same as those in the old one.
    Args:
        base (dict): base dictionary to be added to
        new (dict): new dictionary adding on
    Returns:
        base (dict): updated base dictionary
    """
    for key, val in new.items():
        if key in base:
            base[key] += val
        else:
            base[key] = val
    return base

def main(dset_folder,
         device,
         model_folder,
         batch_size,
         prop,
         sub_batch_size,
         feat_save_folder,
         metric=None,
         test_only=False,
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
      test_only (bool): only load the test set

    """

    # get the model initially by taken the one saved as "best_model"
    model = load_model(model_folder)
    # update its state_dict with the checkpoint from the epoch with
    # the best metric score
    if metric is not None:
        model = model_from_metric(model=model,
                                  model_folder=model_folder,
                                  metric=metric)

    paths, dset_names = get_dset_paths(model_folder, test_only)

    # go through each dataset, create a loader, evaluate the model,
    # and save the predictions

    for i in tqdm(range(len(dset_names))):
        results = {}
        targets = {}
        for path in tqdm(paths[i]):
            dataset = Dataset.from_file(path)
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=collate_dicts)

            new_results, new_targets = evaluate(model,
                                                loader,
                                                device=device,
                                                sub_batch_size=sub_batch_size)

            results = add_dics(base=results, new=new_results)
            targets = add_dics(base=targets, new=new_targets)

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

    fprint("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str,
                        help="Name of model path")
    parser.add_argument('--dset_folder', type=str,
                        help=("Name of the folder with the "
                              "datasets you want to add "
                              "fingerprints to"))
    parser.add_argument('--feat_save_folder', type=str,
                        help="Path to save pickles")
    parser.add_argument('--device', type=str,
                        help="Name of device to use")
    parser.add_argument('--batch_size', type=int,
                        help="Batch size")
    parser.add_argument('--prop', type=str,
                        help="Property to predict",
                        default=None)
    parser.add_argument('--sub_batch_size', type=int,
                        help="Sub batch size",
                        default=None)
    parser.add_argument('--metric', type=str,
                        help=("Select the model with the best validation "
                              "score on this metric. If no metric "
                              "is given, the metric used in the training "
                              "process will be used."),
                        default=None,
                        choices=METRIC_LIST)
    parser.add_argument('--test_only', action='store_true',
                        help=("Only evaluate model "
                              "and generate fingerprints for "
                              "the test set"))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))
    args = parse_args(parser)

    main(**args.__dict__)
