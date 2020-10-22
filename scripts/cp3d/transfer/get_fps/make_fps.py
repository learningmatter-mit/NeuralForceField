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

    if metric in CHEMPROP_TRANSFORM:
        use_metric = CHEMPROP_TRANSFORM[metric]

    else:
        use_metric = metric

    best_score, best_epoch = parse_score(model_folder, use_metric)
    check_path = os.path.join(model_folder, "checkpoints",
                              f"checkpoint-{best_epoch}.pth.tar")

    state_dict = torch.load(check_path, map_location="cpu"
                            )["model"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def fps_and_pred(model, batch, **kwargs):

    model.eval()

    outputs, xyz = model.make_embeddings(batch, xyz=None, **kwargs)
    pooled_fp, learned_weights = model.pool(outputs)
    results = model.readout(pooled_fp)
    results = model.add_grad(batch=batch, results=results, xyz=xyz)

    conf_fps = [i.cpu().detach() for i in outputs["conf_fps_by_smiles"]]
    energy = batch.get("energy")
    boltz_weights = batch.get("weights")

    results.update({"fp": pooled_fp,
                    "conf_fps": conf_fps,
                    "learned_weights": learned_weights,
                    "energy": energy,
                    "boltz_weights": boltz_weights})
    return results


def evaluate(model,
             loader,
             device,
             **kwargs):

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


def get_dsets(full_path, test_only):
    if test_only:
        dset_names = ['test']
    else:
        dset_names = ["train", "val", "test"]

    dsets = []
    for name in tqdm(dset_names):
        new_path = os.path.join(full_path, "{}.pth.tar".format(name))
        dsets.append(Dataset.from_file(new_path))

    return dsets, dset_names


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

    model = load_model(model_folder)
    if metric is not None:
        model = model_from_metric(model=model,
                                  model_folder=model_folder,
                                  metric=metric)
    datasets, dset_names = get_dsets(dset_folder, test_only)

    for i in tqdm(range(len(dset_names))):
        dataset = datasets[i]
        name = dset_names[i]
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_dicts)

        results, targets = evaluate(model,
                                    loader,
                                    device=device,
                                    sub_batch_size=sub_batch_size)

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
